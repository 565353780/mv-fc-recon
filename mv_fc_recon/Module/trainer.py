import os
import torch
import trimesh
import numpy as np
from tqdm import tqdm
from typing import Union, List, Dict, Optional
from torch.utils.tensorboard import SummaryWriter

from camera_control.Module.rgbd_camera import RGBDCamera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from mv_fc_recon.Loss.sdf_reg import computeSDFRegLoss
from mv_fc_recon.Loss.eikonal import eikonal_loss
from mv_fc_recon.Loss.curvature import curvature_loss, get_curvature_weight
from mv_fc_recon.Loss.finite_difference import (
    compute_sdf_gradients,
    trilinear_interpolate_sdf,
)
from mv_fc_recon.Loss.flexicubes_reg import (
    sdf_smoothness_loss,
    weight_regularization_loss,
    mesh_laplacian_smoothness_loss,
    mesh_normal_consistency_loss,
    compute_flexicubes_regularization,
)
from mv_fc_recon.Module.fc_convertor import FCConvertor


class Trainer(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def createOptimizer(
        fc_params: Dict,
        lr: float = 0.01,
        lr_sdf: float = None,
        lr_deform: float = None,
        lr_weight: float = None,
    ) -> torch.optim.Adam:
        """
        为FlexiCubes参数创建优化器

        Args:
            fc_params: createFC返回的参数字典
            lr: 默认学习率
            lr_sdf: SDF学习率（可选）
            lr_deform: 变形学习率（可选）
            lr_weight: 权重学习率（可选）

        Returns:
            optimizer: Adam优化器
        """
        if lr_sdf is None:
            lr_sdf = lr
        if lr_deform is None:
            lr_deform = lr
        if lr_weight is None:
            lr_weight = lr

        param_groups = [
            {'params': [fc_params['sdf']], 'lr': lr_sdf},
            {'params': [fc_params['deform']], 'lr': lr_deform},
            {'params': [fc_params['weight']], 'lr': lr_weight},
        ]

        optimizer = torch.optim.Adam(param_groups)
        return optimizer

    @staticmethod
    def fitImages(
        camera_list: List[RGBDCamera],
        mesh: Union[str, trimesh.Trimesh, None] = None,
        resolution: int = 64,
        device: str = 'cuda:0',
        bg_color: list = [255, 255, 255],
        num_iterations: int = 1000,
        lr: float = 1e-3,
        lambda_dev: float = 0.5,
        lambda_sdf_reg: float = 0.2,
        log_interval: int = 10,
        log_dir: str = './output/',
    ) -> trimesh.Trimesh:
        """
        通过多视角图像拟合FlexiCubes参数

        参考: https://github.com/nv-tlabs/FlexiCubes/blob/main/examples/optimization.ipynb

        Args:
            camera_list: 相机列表，每个相机应包含rgb_image属性
            mesh: 初始网格（可选），如果为None则随机初始化
            resolution: FlexiCubes分辨率
            device: 计算设备
            bg_color: 背景颜色
            num_iterations: 迭代次数
            lr: 学习率
            lambda_dev: developability正则化权重
            lambda_sdf_reg: SDF边缘正则化权重
            log_interval: 日志打印间隔
            log_dir: TensorBoard日志目录，如果为None则不记录

        Returns:
            拟合后的mesh
        """
        # 创建FlexiCubes参数
        fc_params = FCConvertor.createFC(mesh, resolution, device)
        if fc_params is None:
            return None

        # 将相机移动到指定设备并预处理目标图像
        target_images = []
        for camera in camera_list:
            camera.to(device=device)
            # 预处理目标图像
            target_rgb = camera.image
            target_images.append(target_rgb)

        # 创建优化器
        optimizer = Trainer.createOptimizer(fc_params, lr=lr)

        # 获取grid_edges用于SDF正则化
        grid_edges = fc_params['grid_edges']

        # 创建TensorBoard writer
        writer = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
            # 记录GT RGB图像（只记录一次）
            for i, target_rgb in enumerate(target_images):
                writer.add_image(f'GT/Camera_{i}', target_rgb.clone().permute(2, 0, 1), global_step=0)

        if log_dir:
            with torch.no_grad():
                # 从FlexiCubes参数提取mesh（只提取一次，用于所有视角）
                curr_mesh, _, _ = FCConvertor.extractMesh(
                    fc_params, training=False
                )
                # 检查网格是否有效再导出
                if curr_mesh is not None and len(curr_mesh.vertices) > 0 and len(curr_mesh.faces) > 0:
                    try:
                        curr_mesh.export(log_dir + 'start_fc_mesh.ply')
                    except Exception as e:
                        print(f'[WARNING] Failed to export start mesh: {e}')
                else:
                    print('[WARNING] Start mesh is empty, skipping export')

        # 训练循环
        pbar = tqdm(range(num_iterations), desc='FlexiCubes Optimization')
        for iteration in pbar:
            optimizer.zero_grad()

            try:
                # 从FlexiCubes参数提取mesh（只提取一次，用于所有视角）
                current_mesh, vertices, L_dev = FCConvertor.extractMesh(
                    fc_params, training=True
                )

                # 检查网格有效性
                if current_mesh is None or len(current_mesh.vertices) == 0 or len(current_mesh.faces) == 0:
                    print(f'[WARNING] Invalid mesh at iteration {iteration}, skipping...')
                    continue

                # 检查顶点和面片数量是否合理
                if len(current_mesh.vertices) > 1000000 or len(current_mesh.faces) > 2000000:
                    print(f'[WARNING] Mesh too large at iteration {iteration}, skipping...')
                    continue

                # 使用所有相机进行渲染
                num_cameras = len(camera_list)
                batch_indices = list(range(num_cameras))

                total_render_loss = 0.0
                render_rgb_list = []  # 保存前4个视角的渲染结果用于TensorBoard
                render_idx_list = []  # 保存对应的索引

                for idx in batch_indices:
                    camera = camera_list[idx]
                    target_rgb = target_images[idx]

                    # 渲染
                    render_dict = NVDiffRastRenderer.renderVertexColor(
                        mesh=current_mesh,
                        camera=camera,
                        bg_color=bg_color,
                        vertices_tensor=vertices,
                        enable_antialias=True,
                    )

                    render_rgb = render_dict['image']  # [H, W, 3]

                    # 确保渲染结果在[0, 1]范围
                    if render_rgb.max() > 1.0:
                        render_rgb = render_rgb / 255.0

                    # 保存前4个视角的渲染结果用于TensorBoard
                    if len(render_rgb_list) < 4:
                        render_rgb_list.append(render_rgb.clone())
                        render_idx_list.append(idx)

                    # RGB loss（只在有效区域计算）
                    rgb_loss = ((render_rgb - target_rgb).abs()).mean()

                    # 组合渲染损失
                    total_render_loss = total_render_loss + rgb_loss

                # 平均渲染损失
                avg_render_loss = total_render_loss / len(batch_indices)

                # FlexiCubes developability正则化损失
                loss_dev = L_dev.mean() if L_dev is not None and L_dev.numel() > 0 else torch.tensor(0.0, device=device)

                # SDF边缘正则化损失
                loss_sdf_reg = computeSDFRegLoss(fc_params['sdf'], grid_edges)

                # 总损失
                total_loss = avg_render_loss + lambda_dev * loss_dev + lambda_sdf_reg * loss_sdf_reg

                # 检查损失是否包含NaN或Inf
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f'[WARNING] Invalid loss (NaN/Inf) at iteration {iteration}, skipping...')
                    continue

                # 反向传播
                total_loss.backward()

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(
                    [fc_params['sdf'], fc_params['deform'], fc_params['weight']],
                    max_norm=1.0
                )

                # 检查梯度是否包含NaN或Inf
                has_nan_grad = False
                for param in [fc_params['sdf'], fc_params['deform'], fc_params['weight']]:
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break
                
                if has_nan_grad:
                    print(f'[WARNING] NaN/Inf gradients at iteration {iteration}, skipping update...')
                    optimizer.zero_grad()
                    continue

                # 更新参数
                optimizer.step()

                # 裁剪SDF值，防止极端值导致网格提取失败
                with torch.no_grad():
                    fc_params['sdf'].data = torch.clamp(fc_params['sdf'].data, -10.0, 10.0)

            except Exception as e:
                print(f'[ERROR] Exception at iteration {iteration}: {e}')
                # 如果出现异常，尝试恢复参数到上一个有效状态
                # 这里我们简单地跳过这次迭代
                optimizer.zero_grad()
                continue

            # 更新进度条
            if iteration % log_interval == 0 or iteration == num_iterations - 1:
                # 记录到TensorBoard
                if writer is not None:
                    # 记录所有loss
                    writer.add_scalar('Loss/Total', total_loss.item(), iteration)
                    writer.add_scalar('Loss/Render', avg_render_loss.item(), iteration)
                    writer.add_scalar('Loss/Dev', loss_dev.item(), iteration)
                    writer.add_scalar('Loss/SDF_Reg', loss_sdf_reg.item(), iteration)

                    # 记录渲染的RGB图像（只保存前4张）
                    for i, (render_rgb, render_idx) in enumerate(zip(render_rgb_list, render_idx_list)):
                        if render_rgb.dim() == 3 and render_rgb.shape[-1] == 3:
                            render_rgb = render_rgb.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
                        writer.add_image(f'Render/Camera_{render_idx}', render_rgb, global_step=iteration)

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'render': f'{avg_render_loss.item():.4f}',
                'dev': f'{loss_dev.item():.4f}',
                'sdf_reg': f'{loss_sdf_reg.item():.4f}',
            })

        # 关闭TensorBoard writer
        if writer is not None:
            writer.close()

        # 提取最终mesh
        final_mesh, _, _ = FCConvertor.extractMesh(fc_params, training=False)

        return final_mesh

    @staticmethod
    def fitDepthImages(
        camera_list: List[RGBDCamera],
        mesh: Union[str, trimesh.Trimesh, None] = None,
        resolution: int = 64,
        device: str = 'cuda:0',
        bg_color: list = [255, 255, 255],
        num_iterations: int = 1000,
        batch_size: int = 8,
        lr: float = 0.01,
        lambda_dev: float = 0.5,
        lambda_sdf_reg: float = 0.2,
        lambda_depth: float = 1.0,
        log_interval: int = 1,
    ) -> trimesh.Trimesh:
        """
        通过多视角深度图像拟合FlexiCubes参数

        参考: https://github.com/nv-tlabs/FlexiCubes/blob/main/examples/optimization.ipynb

        Args:
            camera_list: 相机列表，每个相机应包含depth_image属性
            mesh: 初始网格（可选），如果为None则随机初始化
            resolution: FlexiCubes分辨率
            device: 计算设备
            bg_color: 背景颜色
            num_iterations: 迭代次数
            batch_size: 每次迭代采样的视角数量
            lr: 学习率
            lambda_dev: developability正则化权重
            lambda_sdf_reg: SDF边缘正则化权重
            lambda_depth: 深度损失权重
            log_interval: 日志打印间隔

        Returns:
            拟合后的mesh
        """
        # 创建FlexiCubes参数
        fc_params = FCConvertor.createFC(mesh, resolution, device)
        if fc_params is None:
            return None

        # 将相机移动到指定设备并预处理目标深度图
        target_depths = []
        target_masks = []
        for camera in camera_list:
            camera.to(device=device)
            # 预处理目标深度图
            target_depth = camera.depth_image
            if isinstance(target_depth, np.ndarray):
                target_depth = torch.tensor(target_depth, dtype=torch.float32, device=device)
            # 创建有效深度mask
            valid_mask = (target_depth > 0).float()
            target_depths.append(target_depth)
            target_masks.append(valid_mask)

        # 创建优化器
        optimizer = Trainer.createOptimizer(fc_params, lr=lr)

        # 获取grid_edges用于SDF正则化
        grid_edges = fc_params['grid_edges']

        # 训练循环
        pbar = tqdm(range(num_iterations), desc='FlexiCubes Depth Optimization')
        for iteration in pbar:
            optimizer.zero_grad()

            try:
                # 从FlexiCubes参数提取mesh
                current_mesh, vertices, L_dev = FCConvertor.extractMesh(
                    fc_params, training=True
                )

                # 检查网格有效性
                if current_mesh is None or len(current_mesh.vertices) == 0 or len(current_mesh.faces) == 0:
                    print(f'[WARNING] Invalid mesh at iteration {iteration}, skipping...')
                    continue

                # 检查顶点和面片数量是否合理
                if len(current_mesh.vertices) > 1000000 or len(current_mesh.faces) > 2000000:
                    print(f'[WARNING] Mesh too large at iteration {iteration}, skipping...')
                    continue

                # 随机采样batch_size个视角
                num_cameras = len(camera_list)
                if batch_size >= num_cameras:
                    batch_indices = list(range(num_cameras))
                else:
                    batch_indices = np.random.choice(num_cameras, batch_size, replace=False).tolist()

                total_depth_loss = 0.0
                total_mask_loss = 0.0

                for idx in batch_indices:
                    camera = camera_list[idx]
                    target_depth = target_depths[idx]
                    target_mask = target_masks[idx]

                    # 渲染深度
                    render_dict = NVDiffRastRenderer.renderDepth(
                        mesh=current_mesh,
                        camera=camera,
                        bg_color=bg_color,
                        vertices_tensor=vertices,
                        enable_antialias=True,
                    )

                    render_depth = render_dict['depth']  # [H, W]
                    rasterize_output = render_dict['rasterize_output']  # [H, W, 4]

                    # 渲染的mask
                    render_mask = (rasterize_output[..., 3] > 0).float()  # [H, W]

                    # Mask loss
                    intersection = (render_mask * target_mask).sum()
                    union = render_mask.sum() + target_mask.sum() - intersection
                    mask_loss = 1.0 - intersection / (union + 1e-8)

                    # Depth loss（只在有效区域计算）
                    valid_mask = render_mask * target_mask
                    depth_diff = (render_depth - target_depth).abs() * valid_mask
                    depth_loss = depth_diff.sum() / (valid_mask.sum() + 1e-8)

                    total_depth_loss += depth_loss
                    total_mask_loss += mask_loss

                # 平均损失
                avg_depth_loss = total_depth_loss / len(batch_indices)
                avg_mask_loss = total_mask_loss / len(batch_indices)

                # FlexiCubes developability正则化损失
                loss_dev = L_dev.mean() if L_dev is not None and L_dev.numel() > 0 else torch.tensor(0.0, device=device)

                # SDF边缘正则化损失
                loss_sdf_reg = computeSDFRegLoss(fc_params['sdf'], grid_edges)

                # 总损失
                total_loss = (
                    avg_mask_loss +
                    lambda_depth * avg_depth_loss +
                    lambda_dev * loss_dev +
                    lambda_sdf_reg * loss_sdf_reg
                )

                # 检查损失是否包含NaN或Inf
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f'[WARNING] Invalid loss (NaN/Inf) at iteration {iteration}, skipping...')
                    continue

                # 反向传播
                total_loss.backward()

                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(
                    [fc_params['sdf'], fc_params['deform'], fc_params['weight']],
                    max_norm=1.0
                )

                # 检查梯度是否包含NaN或Inf
                has_nan_grad = False
                for param in [fc_params['sdf'], fc_params['deform'], fc_params['weight']]:
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break
                
                if has_nan_grad:
                    print(f'[WARNING] NaN/Inf gradients at iteration {iteration}, skipping update...')
                    optimizer.zero_grad()
                    continue

                # 更新参数
                optimizer.step()

                # 裁剪SDF值，防止极端值导致网格提取失败
                with torch.no_grad():
                    fc_params['sdf'].data = torch.clamp(fc_params['sdf'].data, -10.0, 10.0)

            except Exception as e:
                print(f'[ERROR] Exception at iteration {iteration}: {e}')
                # 如果出现异常，尝试恢复参数到上一个有效状态
                # 这里我们简单地跳过这次迭代
                optimizer.zero_grad()
                continue

            # 更新进度条
            if iteration % log_interval == 0 or iteration == num_iterations - 1:
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'mask': f'{avg_mask_loss.item():.4f}',
                    'depth': f'{avg_depth_loss.item():.4f}',
                    'dev': f'{loss_dev.item():.4f}',
                })

        # 提取最终mesh
        final_mesh, _, _ = FCConvertor.extractMesh(fc_params, training=False)

        return final_mesh

    @staticmethod
    def sampleSDFPoints(
        fc_params: Dict,
        num_samples: int = 1024,
        near_surface_ratio: float = 0.7,  # 从 0.5 增加到 0.7，更多近表面采样
        surface_std: float = 0.005,      # 从 0.01 降低到 0.005，更靠近表面
    ) -> torch.Tensor:
        """在 SDF 场中采样点用于计算 eikonal 和 curvature loss

        Args:
            fc_params: FlexiCubes 参数字典
            num_samples: 总采样点数
            near_surface_ratio: 近表面采样点比例
            surface_std: 近表面采样的标准差

        Returns:
            points: [num_samples, 3] 采样点坐标
        """
        device = fc_params['sdf'].device
        resolution = fc_params['resolution']

        # 获取 SDF 网格
        sdf = fc_params['sdf']
        x_nx3 = fc_params['x_nx3']

        # 近表面采样：在 SDF 接近 0 的点附近采样
        num_near_surface = int(num_samples * near_surface_ratio)
        num_uniform = num_samples - num_near_surface

        # 找到 SDF 接近 0 的点（表面附近）
        surface_mask = sdf.abs() < 0.1  # 阈值可调
        if surface_mask.sum() > 0:
            surface_points = x_nx3[surface_mask]
            # 随机选择一些表面点
            num_select = min(num_near_surface, surface_points.shape[0])
            indices = torch.randperm(surface_points.shape[0], device=device)[:num_select]
            base_points = surface_points[indices]
            # 添加高斯噪声
            noise = torch.randn_like(base_points) * surface_std
            near_surface_points = base_points + noise
            # 裁剪到 [-1, 1] 范围
            near_surface_points = torch.clamp(near_surface_points, -1.0, 1.0)
        else:
            # 如果没有表面点，使用均匀采样
            near_surface_points = torch.rand(num_near_surface, 3, device=device) * 2 - 1

        # 均匀采样
        uniform_points = torch.rand(num_uniform, 3, device=device) * 2 - 1

        # 合并采样点
        points = torch.cat([near_surface_points, uniform_points], dim=0)

        return points

    @staticmethod
    def computeSDFLosses(
        fc_params: Dict,
        sample_points: Optional[torch.Tensor] = None,
        num_samples: int = 1024,
        eps: Optional[float] = None,
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """计算 SDF 相关的 loss（eikonal 和 curvature）

        参考: neural-angelo/neural_angelo/Module/trainer.py

        Args:
            fc_params: FlexiCubes 参数字典
            sample_points: [M, 3] 采样点坐标，如果为 None 则自动采样
            num_samples: 自动采样时的采样点数
            eps: 有限差分步长
            training: 是否处于训练模式

        Returns:
            losses: 包含 'eikonal' 和 'curvature' 的字典
        """
        device = fc_params['sdf'].device
        resolution = fc_params['resolution']
        sdf_grid = fc_params['sdf']
        x_nx3 = fc_params['x_nx3']

        # 默认步长为网格间距（使用更小的步长以提高精度）
        if eps is None:
            eps = 1.0 / resolution  # 从 2.0/resolution 改为 1.0/resolution

        # 采样点
        if sample_points is None:
            sample_points = Trainer.sampleSDFPoints(fc_params, num_samples)

        # 创建 SDF 插值函数
        def sdf_func(points):
            return trilinear_interpolate_sdf(sdf_grid, x_nx3, points, resolution)

        # 计算梯度和 Hessian
        gradients, hessians = compute_sdf_gradients(
            sdf_func=sdf_func,
            x=sample_points,
            eps=eps,
            taps=6,
            training=training,
        )

        # 计算 losses
        losses = {}
        losses['eikonal'] = eikonal_loss(gradients)
        losses['curvature'] = curvature_loss(hessians) if training else torch.tensor(0.0, device=device)

        return losses

    @staticmethod
    def fitImagesWithSDFLoss(
        camera_list: List[RGBDCamera],
        mesh: Union[str, trimesh.Trimesh, None] = None,
        resolution: int = 128,
        device: str = 'cuda:0',
        bg_color: list = [255, 255, 255],
        num_iterations: int = 1000,
        lr: float = 5e-4,
        # 渲染权重（主要驱动力，引导网格变化）
        lambda_render: float = 1.0,         # 渲染损失权重
        # FlexiCubes 专用正则化（推荐使用）
        lambda_sdf_smooth: float = 0.1,     # SDF 平滑：惩罚相邻网格点 SDF 差异
        lambda_weight_reg: float = 0.01,    # 权重正则化：约束 alpha/beta/gamma
        lambda_mesh_smooth: float = 0.0,    # 网格 Laplacian 平滑（可选）
        lambda_normal_consistency: float = 0.0,  # 法线一致性（可选）
        lambda_dev: float = 0.5,            # FlexiCubes 可展性正则化
        # 传统 SDF 约束（不推荐用于 FlexiCubes，默认关闭）
        lambda_eikonal: float = 0.0,        # Eikonal 约束：不适用于 FlexiCubes！
        lambda_curvature: float = 0.0,      # 曲率约束：不适用于 FlexiCubes！
        lambda_sdf_reg: float = 0.0,        # 原始 SDF 边缘约束（BCE 版本，不稳定）
        # 其他参数
        num_sdf_samples: int = 1024,        # SDF 采样点数（仅用于传统 SDF 约束）
        log_interval: int = 10,
        log_dir: str = './output/',
    ) -> trimesh.Trimesh:
        """通过多视角图像拟合 FlexiCubes 参数

        ⚠️ 重要说明：FlexiCubes 的 SDF 与传统 SDF（NeuS、NeuralAngelo）有本质区别！

        FlexiCubes 的 SDF 特点：
        1. 只需要符号正确（正=外部，负=内部），不需要是真正的距离场
        2. SDF 值的尺度是任意的，不需要满足 ||∇SDF|| = 1
        3. 表面位置由 SDF 零交叉点决定，而非 SDF 值本身

        因此：
        - ❌ Eikonal loss（强制 ||∇SDF|| = 1）会导致 SDF 振荡，产生噪声
        - ❌ 基于 Hessian 的 curvature loss 在离散网格上会放大噪声
        - ✅ SDF 平滑正则化：惩罚相邻网格点 SDF 差异过大
        - ✅ 权重正则化：约束 FlexiCubes 的 alpha/beta/gamma 权重
        - ✅ 网格 Laplacian 平滑：直接在提取的网格顶点上平滑
        - ✅ 法线一致性：惩罚相邻面片法线差异过大

        训练策略：
        - Render loss 作为主要驱动力，引导网格向目标图像对齐
        - FlexiCubes 专用正则化确保表面平滑

        Args:
            camera_list: 相机列表，每个相机应包含 rgb_image 属性
            mesh: 初始网格（可选），如果为 None 则随机初始化
            resolution: FlexiCubes 分辨率
            device: 计算设备
            bg_color: 背景颜色
            num_iterations: 迭代次数
            lr: 学习率
            lambda_render: 渲染损失权重（主要驱动力）
            lambda_sdf_smooth: SDF 平滑权重（推荐 0.1-0.5）
            lambda_weight_reg: 权重正则化权重（推荐 0.01-0.1）
            lambda_mesh_smooth: 网格 Laplacian 平滑权重（可选，推荐 0.0-0.1）
            lambda_normal_consistency: 法线一致性权重（可选，推荐 0.0-0.1）
            lambda_dev: FlexiCubes developability 正则化权重
            lambda_eikonal: eikonal loss 权重（⚠️ 不推荐用于 FlexiCubes，默认 0）
            lambda_curvature: curvature loss 权重（⚠️ 不推荐用于 FlexiCubes，默认 0）
            lambda_sdf_reg: 原始 SDF 边缘正则化权重（⚠️ BCE 版本不稳定，默认 0）
            num_sdf_samples: SDF 采样点数（仅用于传统 SDF 约束）
            log_interval: 日志打印间隔
            log_dir: TensorBoard 日志目录

        Returns:
            拟合后的 mesh
        """
        # 警告：如果用户启用了不适合 FlexiCubes 的 loss
        if lambda_eikonal > 0:
            print('[WARNING] Eikonal loss is NOT suitable for FlexiCubes!')
            print('         FlexiCubes SDF does not need ||∇SDF|| = 1.')
            print('         This may cause noisy surfaces. Consider setting lambda_eikonal=0.')
        if lambda_curvature > 0:
            print('[WARNING] Hessian-based curvature loss is NOT suitable for FlexiCubes!')
            print('         This may amplify noise on discrete grids. Consider setting lambda_curvature=0.')
        if lambda_sdf_reg > 0:
            print('[WARNING] BCE-based SDF regularization may be unstable.')
            print('         Consider using lambda_sdf_smooth instead.')

        # 创建 FlexiCubes 参数
        fc_params = FCConvertor.createFC(mesh, resolution, device)
        if fc_params is None:
            return None

        # 将相机移动到指定设备并预处理目标图像
        target_images = []
        for camera in camera_list:
            camera.to(device=device)
            target_rgb = camera.image
            target_images.append(target_rgb)

        # 创建优化器
        optimizer = Trainer.createOptimizer(fc_params, lr=lr)

        # 获取 grid_edges 用于 SDF 正则化
        grid_edges = fc_params['grid_edges']

        # 创建 TensorBoard writer
        writer = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
            for i, target_rgb in enumerate(target_images):
                writer.add_image(f'GT/Camera_{i}', target_rgb.clone().permute(2, 0, 1), global_step=0)

        if log_dir:
            with torch.no_grad():
                curr_mesh, _, _ = FCConvertor.extractMesh(fc_params, training=False)
                if curr_mesh is not None and len(curr_mesh.vertices) > 0 and len(curr_mesh.faces) > 0:
                    try:
                        curr_mesh.export(log_dir + 'start_fc_mesh.ply')
                    except Exception as e:
                        print(f'[WARNING] Failed to export start mesh: {e}')

        # 训练循环
        pbar = tqdm(range(num_iterations), desc='FlexiCubes Optimization')
        for iteration in pbar:
            optimizer.zero_grad()

            try:
                # 从 FlexiCubes 参数提取 mesh
                current_mesh, vertices, L_dev = FCConvertor.extractMesh(fc_params, training=True)

                # 检查网格有效性
                if current_mesh is None or len(current_mesh.vertices) == 0 or len(current_mesh.faces) == 0:
                    print(f'[WARNING] Invalid mesh at iteration {iteration}, skipping...')
                    continue

                if len(current_mesh.vertices) > 1000000 or len(current_mesh.faces) > 2000000:
                    print(f'[WARNING] Mesh too large at iteration {iteration}, skipping...')
                    continue

                # 获取 faces tensor 用于网格平滑
                faces_tensor = torch.from_numpy(current_mesh.faces).long().to(device)

                # ========== 渲染损失 ==========
                total_render_loss = 0.0
                render_rgb_list = []
                render_idx_list = []

                if lambda_render > 0:
                    num_cameras = len(camera_list)
                    batch_indices = list(range(num_cameras))

                    for idx in batch_indices:
                        camera = camera_list[idx]
                        target_rgb = target_images[idx]

                        render_dict = NVDiffRastRenderer.renderVertexColor(
                            mesh=current_mesh,
                            camera=camera,
                            bg_color=bg_color,
                            vertices_tensor=vertices,
                            enable_antialias=True,
                        )

                        render_rgb = render_dict['image']
                        if render_rgb.max() > 1.0:
                            render_rgb = render_rgb / 255.0

                        if len(render_rgb_list) < 4:
                            render_rgb_list.append(render_rgb.clone())
                            render_idx_list.append(idx)

                        rgb_loss = ((render_rgb - target_rgb).abs()).mean()
                        total_render_loss = total_render_loss + rgb_loss

                    avg_render_loss = total_render_loss / len(batch_indices)
                else:
                    avg_render_loss = torch.tensor(0.0, device=device)

                # ========== FlexiCubes 专用正则化（推荐） ==========

                # FlexiCubes developability 正则化损失
                loss_dev = L_dev.mean() if L_dev is not None and L_dev.numel() > 0 else torch.tensor(0.0, device=device)

                # SDF 平滑损失：惩罚相邻网格点 SDF 差异过大
                if lambda_sdf_smooth > 0:
                    loss_sdf_smooth = sdf_smoothness_loss(fc_params['sdf'], grid_edges)
                else:
                    loss_sdf_smooth = torch.tensor(0.0, device=device)

                # 权重正则化损失：约束 alpha/beta/gamma 权重
                if lambda_weight_reg > 0:
                    loss_weight_reg = weight_regularization_loss(fc_params['weight'])
                else:
                    loss_weight_reg = torch.tensor(0.0, device=device)

                # 网格 Laplacian 平滑损失
                if lambda_mesh_smooth > 0:
                    loss_mesh_smooth = mesh_laplacian_smoothness_loss(vertices, faces_tensor)
                else:
                    loss_mesh_smooth = torch.tensor(0.0, device=device)

                # 法线一致性损失
                if lambda_normal_consistency > 0:
                    loss_normal_consistency = mesh_normal_consistency_loss(vertices, faces_tensor)
                else:
                    loss_normal_consistency = torch.tensor(0.0, device=device)

                # ========== 传统 SDF 约束（不推荐用于 FlexiCubes） ==========

                # 原始 SDF 边缘正则化损失（BCE 版本）
                if lambda_sdf_reg > 0:
                    loss_sdf_reg = computeSDFRegLoss(fc_params['sdf'], grid_edges)
                else:
                    loss_sdf_reg = torch.tensor(0.0, device=device)

                # Eikonal 和 Curvature loss（不推荐）
                if lambda_eikonal > 0 or lambda_curvature > 0:
                    sdf_losses = Trainer.computeSDFLosses(
                        fc_params,
                        num_samples=num_sdf_samples,
                        training=True,
                    )
                    loss_eikonal = sdf_losses['eikonal'] if lambda_eikonal > 0 else torch.tensor(0.0, device=device)
                    loss_curvature = sdf_losses['curvature'] if lambda_curvature > 0 else torch.tensor(0.0, device=device)
                else:
                    loss_eikonal = torch.tensor(0.0, device=device)
                    loss_curvature = torch.tensor(0.0, device=device)

                # ========== 总损失 ==========
                total_loss = (
                    # 渲染损失（主要驱动力）
                    lambda_render * avg_render_loss +
                    # FlexiCubes 专用正则化（推荐）
                    lambda_dev * loss_dev +
                    lambda_sdf_smooth * loss_sdf_smooth +
                    lambda_weight_reg * loss_weight_reg +
                    lambda_mesh_smooth * loss_mesh_smooth +
                    lambda_normal_consistency * loss_normal_consistency +
                    # 传统 SDF 约束（不推荐）
                    lambda_sdf_reg * loss_sdf_reg +
                    lambda_eikonal * loss_eikonal +
                    lambda_curvature * loss_curvature
                )

                # 检查损失是否包含 NaN 或 Inf
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f'[WARNING] Invalid loss (NaN/Inf) at iteration {iteration}, skipping...')
                    continue

                # 反向传播
                total_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(
                    [fc_params['sdf'], fc_params['deform'], fc_params['weight']],
                    max_norm=1.0
                )

                # 检查梯度是否包含 NaN 或 Inf
                has_nan_grad = False
                for param in [fc_params['sdf'], fc_params['deform'], fc_params['weight']]:
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            has_nan_grad = True
                            break

                if has_nan_grad:
                    print(f'[WARNING] NaN/Inf gradients at iteration {iteration}, skipping update...')
                    optimizer.zero_grad()
                    continue

                # 更新参数
                optimizer.step()

                # 裁剪 SDF 值
                with torch.no_grad():
                    fc_params['sdf'].data = torch.clamp(fc_params['sdf'].data, -10.0, 10.0)

            except Exception as e:
                print(f'[ERROR] Exception at iteration {iteration}: {e}')
                optimizer.zero_grad()
                continue

            # 更新进度条和日志
            if iteration % log_interval == 0 or iteration == num_iterations - 1:
                if writer is not None:
                    writer.add_scalar('Loss/Total', total_loss.item(), iteration)
                    writer.add_scalar('Loss/Render', avg_render_loss.item(), iteration)
                    writer.add_scalar('Loss/Dev', loss_dev.item(), iteration)
                    writer.add_scalar('Loss/SDF_Smooth', loss_sdf_smooth.item(), iteration)
                    writer.add_scalar('Loss/Weight_Reg', loss_weight_reg.item(), iteration)
                    if lambda_mesh_smooth > 0:
                        writer.add_scalar('Loss/Mesh_Smooth', loss_mesh_smooth.item(), iteration)
                    if lambda_normal_consistency > 0:
                        writer.add_scalar('Loss/Normal_Consistency', loss_normal_consistency.item(), iteration)
                    if lambda_sdf_reg > 0:
                        writer.add_scalar('Loss/SDF_Reg', loss_sdf_reg.item(), iteration)
                    if lambda_eikonal > 0:
                        writer.add_scalar('Loss/Eikonal', loss_eikonal.item(), iteration)
                    if lambda_curvature > 0:
                        writer.add_scalar('Loss/Curvature', loss_curvature.item(), iteration)

                    for i, (render_rgb, render_idx) in enumerate(zip(render_rgb_list, render_idx_list)):
                        if render_rgb.dim() == 3 and render_rgb.shape[-1] == 3:
                            render_rgb = render_rgb.permute(2, 0, 1)
                        writer.add_image(f'Render/Camera_{render_idx}', render_rgb, global_step=iteration)

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'render': f'{avg_render_loss.item():.4f}',
                'sdf_sm': f'{loss_sdf_smooth.item():.4f}',
                'dev': f'{loss_dev.item():.4f}',
            })

        # 关闭 TensorBoard writer
        if writer is not None:
            writer.close()

        # 提取最终 mesh
        final_mesh, _, _ = FCConvertor.extractMesh(fc_params, training=False)

        return final_mesh
