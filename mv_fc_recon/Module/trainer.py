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
        os.makedirs(log_dir + 'rgb/', exist_ok=True)
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
        near_surface_ratio: float = 0.5,
        surface_std: float = 0.01,
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

        # 默认步长为网格间距
        if eps is None:
            eps = 2.0 / resolution

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
        resolution: int = 64,
        device: str = 'cuda:0',
        bg_color: list = [255, 255, 255],
        num_iterations: int = 1000,
        lr: float = 1e-3,
        lambda_dev: float = 0.5,
        lambda_sdf_reg: float = 0.2,
        lambda_eikonal: float = 0.1,
        lambda_curvature: float = 5e-4,
        warm_up_end: int = 500,
        num_sdf_samples: int = 1024,
        log_interval: int = 10,
        log_dir: str = './output/',
    ) -> trimesh.Trimesh:
        """通过多视角图像拟合 FlexiCubes 参数，包含 eikonal 和 curvature loss

        参考: neural-angelo/neural_angelo/Module/trainer.py

        Args:
            camera_list: 相机列表，每个相机应包含 rgb_image 属性
            mesh: 初始网格（可选），如果为 None 则随机初始化
            resolution: FlexiCubes 分辨率
            device: 计算设备
            bg_color: 背景颜色
            num_iterations: 迭代次数
            lr: 学习率
            lambda_dev: developability 正则化权重
            lambda_sdf_reg: SDF 边缘正则化权重
            lambda_eikonal: eikonal loss 权重
            lambda_curvature: curvature loss 初始权重
            warm_up_end: curvature loss warm-up 结束迭代数
            num_sdf_samples: SDF 采样点数
            log_interval: 日志打印间隔
            log_dir: TensorBoard 日志目录

        Returns:
            拟合后的 mesh
        """
        # 创建 FlexiCubes 参数
        fc_params = FCConvertor.createFC(mesh, resolution, device)
        if fc_params is None:
            return None

        # 将相机移动到指定设备并预处理目标图像
        target_images = []
        os.makedirs(log_dir + 'rgb/', exist_ok=True)
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
        pbar = tqdm(range(num_iterations), desc='FlexiCubes Optimization with SDF Loss')
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

                # 使用所有相机进行渲染
                num_cameras = len(camera_list)
                batch_indices = list(range(num_cameras))

                total_render_loss = 0.0
                render_rgb_list = []
                render_idx_list = []

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

                # FlexiCubes developability 正则化损失
                loss_dev = L_dev.mean() if L_dev is not None and L_dev.numel() > 0 else torch.tensor(0.0, device=device)

                # SDF 边缘正则化损失
                loss_sdf_reg = computeSDFRegLoss(fc_params['sdf'], grid_edges)

                # 计算 eikonal 和 curvature loss
                sdf_losses = Trainer.computeSDFLosses(
                    fc_params,
                    num_samples=num_sdf_samples,
                    training=True,
                )
                loss_eikonal = sdf_losses['eikonal']
                loss_curvature = sdf_losses['curvature']

                # 动态调整 curvature 权重
                current_curvature_weight = get_curvature_weight(
                    current_iteration=iteration,
                    init_weight=lambda_curvature,
                    warm_up_end=warm_up_end,
                )

                # 总损失
                total_loss = (
                    avg_render_loss +
                    lambda_dev * loss_dev +
                    lambda_sdf_reg * loss_sdf_reg +
                    lambda_eikonal * loss_eikonal +
                    current_curvature_weight * loss_curvature
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
                    writer.add_scalar('Loss/SDF_Reg', loss_sdf_reg.item(), iteration)
                    writer.add_scalar('Loss/Eikonal', loss_eikonal.item(), iteration)
                    writer.add_scalar('Loss/Curvature', loss_curvature.item(), iteration)
                    writer.add_scalar('Weight/Curvature', current_curvature_weight, iteration)

                    for i, (render_rgb, render_idx) in enumerate(zip(render_rgb_list, render_idx_list)):
                        if render_rgb.dim() == 3 and render_rgb.shape[-1] == 3:
                            render_rgb = render_rgb.permute(2, 0, 1)
                        writer.add_image(f'Render/Camera_{render_idx}', render_rgb, global_step=iteration)

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'render': f'{avg_render_loss.item():.4f}',
                'eik': f'{loss_eikonal.item():.4f}',
                'curv': f'{loss_curvature.item():.4f}',
            })

        # 关闭 TensorBoard writer
        if writer is not None:
            writer.close()

        # 提取最终 mesh
        final_mesh, _, _ = FCConvertor.extractMesh(fc_params, training=False)

        return final_mesh
