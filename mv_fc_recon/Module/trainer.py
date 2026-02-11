import os
import torch
import trimesh
from tqdm import tqdm
from typing import Union, List, Dict, Optional
from torch.utils.tensorboard import SummaryWriter

from camera_control.Module.camera import Camera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from flexi_cubes.Module.fc_convertor import FCConvertor

from mv_fc_recon.Loss.flexicubes_reg import (
    sdf_smoothness_loss,
    sdf_gradient_smoothness_loss,
    weight_regularization_loss,
    mesh_normal_consistency_loss,
    mesh_bi_laplacian_smoothness_loss,
)
from mv_fc_recon.Loss.mesh_geo_energy import thin_plate_energy


class Trainer(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def createOptimizer(
        fc_params: Dict,
        lr: float = 0.01,
        lr_sdf: Optional[float] = None,
        lr_deform: Optional[float] = None,
        lr_weight: Optional[float] = None,
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
            dict(params=[fc_params['sdf']], lr=lr_sdf),
            dict(params=[fc_params['deform']], lr=lr_deform),
            dict(params=[fc_params['weight']], lr=lr_weight),
        ]

        optimizer = torch.optim.Adam(param_groups)
        return optimizer

    @staticmethod
    def fitImagesWithSDFLoss(
        camera_list: List[Camera],
        mesh: Union[str, trimesh.Trimesh, None] = None,
        resolution: int = 128,
        device: str = 'cuda:0',
        bg_color: list = [255, 255, 255],
        num_iterations: int = 120,
        lr: float = 5e-4,
        # 渲染权重（主要驱动力，引导网格变化）
        lambda_render: float = 1.0,         # 渲染损失权重
        lambda_thin_plate_energy: float = 0.5,  ## 1e-6
        # FlexiCubes 专用正则化（推荐使用）
        lambda_sdf_smooth: float = 0.1,     # SDF 平滑：惩罚相邻网格点 SDF 差异
        lambda_sdf_grad_smooth: float = 0.01,  # SDF 二阶平滑：惩罚梯度变化（更好保持细节）
        lambda_weight_reg: float = 0.01,    # 权重正则化：约束 alpha/beta/gamma
        lambda_mesh_smooth: float = 0.0,    # 网格 Laplacian 平滑（可选）
        lambda_normal_consistency: float = 0.01,  # 法线一致性（可选）
        lambda_dev: float = 0.1,            # FlexiCubes 可展性正则化
        # SDF 平滑参数
        sdf_smooth_mode: str = 'l2',  # 'l2', 'adaptive', 'huber' (推荐 'l2'，adaptive 可能信号太少)
        sdf_smooth_threshold: float = 0,  # adaptive/huber 模式的阈值
        # 其他参数
        log_interval: int = 10,
        log_dir: str = './output/',
    ) -> trimesh.Trimesh:
        """通过多视角图像拟合 FlexiCubes 参数

        ⚠️ 重要说明：FlexiCubes 的 SDF 与传统 SDF（NeuS、NeuralAngelo）有本质区别！
        FlexiCubes 的 SDF 特点：
        1. 只需要符号正确（正=外部，负=内部），不需要是真正的距离场
        2. SDF 值的尺度是任意的，不需要满足 ||∇SDF|| = 1
        3. 表面位置由 SDF 零交叉点决定，而非 SDF 值本身

        SDF 平滑模式说明：
        - 'l2': 简单 L2 平滑，惩罚所有 SDF 差异
        - 'adaptive': 自适应平滑，只惩罚超过阈值的差异（推荐）
        - 'huber': Huber loss，对大差异使用 L1，小差异使用 L2

        Args:
            camera_list: 相机列表
            mesh: 初始网格（可选），如果为 None 则随机初始化
            resolution: FlexiCubes 分辨率
            device: 计算设备
            bg_color: 背景颜色
            num_iterations: 迭代次数
            lr: 学习率
            lambda_render: 渲染损失权重（主要驱动力）
            lambda_sdf_smooth: SDF 平滑权重（推荐 0.1-0.5）
            lambda_sdf_grad_smooth: SDF 二阶平滑权重（推荐 0.0-0.1，用于保持细节）
            lambda_weight_reg: 权重正则化权重（推荐 0.01-0.1）
            lambda_mesh_smooth: 网格 Laplacian 平滑权重（可选，推荐 0.0-0.1）
            lambda_normal_consistency: 法线一致性权重（可选，推荐 0.0-0.1）
            lambda_dev: FlexiCubes developability 正则化权重
            sdf_smooth_mode: SDF 平滑模式 ('l2', 'adaptive', 'huber')
            sdf_smooth_threshold: adaptive/huber 模式的阈值
            sdf_smooth_warmup: 动态调整的 warmup 迭代数
            sdf_smooth_scale: 动态调整的最大缩放因子
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
        target_data_list = []
        for camera in camera_list:
            camera.to(device=device)
            target_normal = camera.normal
            target_data_list.append(target_normal)

        # 创建优化器
        optimizer = Trainer.createOptimizer(fc_params, lr=lr)

        # 获取 grid_edges 用于 SDF 正则化
        grid_edges = fc_params['grid_edges']

        # 创建 TensorBoard writer
        writer = None
        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)
            for i, target_data in enumerate(target_data_list):
                writer.add_image(f'GT/Camera_{i}', target_data.clone().permute(2, 0, 1), global_step=0)

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

            # 从 FlexiCubes 参数提取 mesh
            current_mesh, vertices, L_dev = FCConvertor.extractMesh(fc_params, training=True)

            # 检查网格有效性
            if current_mesh is None or len(current_mesh.vertices) == 0 or len(current_mesh.faces) == 0:
                print(f'[WARNING] Invalid mesh at iteration {iteration}, skipping...')
                continue

            # 获取 faces tensor 用于网格平滑
            faces_tensor = torch.from_numpy(current_mesh.faces).long().to(device)

            # ========== 计算实际使用的 Loss ==========
            # 计算总损失（只包含权重 > 0 的项）
            total_loss = torch.tensor(0.0, device=device)
            loss_dict = {}  # 用于 TensorBoard 记录

            # ========== 渲染损失 ==========
            total_render_loss = 0.0
            render_data_list = []
            render_idx_list = []

            avg_render_loss = torch.tensor(0.0, device=device)
            if lambda_render > 0:
                num_cameras = len(camera_list)
                batch_indices = list(range(num_cameras))

                for idx in batch_indices:

                    camera = camera_list[idx]
                    target_data = target_data_list[idx]

                    render_dict = NVDiffRastRenderer.renderNormal(
                        mesh=current_mesh,
                        camera=camera,
                        bg_color=bg_color,
                        vertices_tensor=vertices,
                    )
                    render_data = render_dict['normal_camera']

                    if len(render_data_list) < 4:
                        render_data_list.append(render_data.clone())
                        render_idx_list.append(idx)

                    render_loss = ((render_data - target_data).abs()).mean()
                    total_render_loss = total_render_loss + render_loss

                avg_render_loss = total_render_loss / len(batch_indices)

                total_loss = total_loss + lambda_render * avg_render_loss
                loss_dict['Render'] = avg_render_loss.item()

            # FlexiCubes developability 正则化损失
            loss_dev = L_dev.mean() if L_dev is not None and L_dev.numel() > 0 else torch.tensor(0.0, device=device)

            ## thin-plate energy（缩放到与 render 成固定比例，作为稳定辅助 loss）
            if lambda_thin_plate_energy > 0:
                thinplate_loss = thin_plate_energy(
                    vertices, faces_tensor
                )
                if lambda_render > 0 and avg_render_loss.numel() > 0:
                    # 目标贡献 = lambda_thin_plate_energy/(lambda_render+lambda_thin_plate_energy)*avg_render_loss
                    # 缩放因子计算全部无梯度，梯度只从 thinplate_loss 反传
                    target_contribution = (lambda_thin_plate_energy / (lambda_render + lambda_thin_plate_energy)) * avg_render_loss.detach()
                    scale = target_contribution / (thinplate_loss.detach() + 1e-8)
                    scaled_thinplate = scale.detach() * thinplate_loss
                    total_loss = total_loss + scaled_thinplate
                else:
                    total_loss = total_loss + lambda_thin_plate_energy * thinplate_loss
                loss_dict['thinPlateE'] = thinplate_loss.item()

            # FlexiCubes developability
            if lambda_dev > 0:
                total_loss = total_loss + lambda_dev * loss_dev
                loss_dict['Dev'] = loss_dev.item()

            # SDF 平滑损失
            if lambda_sdf_smooth > 0:
                loss_sdf_smooth = sdf_smoothness_loss(
                    fc_params['sdf'], grid_edges,
                    mode=sdf_smooth_mode,
                    threshold=sdf_smooth_threshold,
                )
                total_loss = total_loss + lambda_sdf_smooth * loss_sdf_smooth
                loss_dict['SDF_Smooth'] = loss_sdf_smooth.item()

            # SDF 二阶平滑损失
            if lambda_sdf_grad_smooth > 0:
                loss_sdf_grad_smooth = sdf_gradient_smoothness_loss(
                    fc_params['sdf'], grid_edges, fc_params['x_nx3'],
                    mode='local',
                )
                total_loss = total_loss + lambda_sdf_grad_smooth * loss_sdf_grad_smooth
                loss_dict['SDF_Grad_Smooth'] = loss_sdf_grad_smooth.item()

            # 权重正则化损失
            if lambda_weight_reg > 0:
                loss_weight_reg = weight_regularization_loss(fc_params['weight'])
                total_loss = total_loss + lambda_weight_reg * loss_weight_reg
                loss_dict['Weight_Reg'] = loss_weight_reg.item()

            # 网格 Laplacian 平滑损失
            # 不如优化图像上 当前渲染的normal map rgb图的光滑程度？
            if lambda_mesh_smooth > 0 and iteration >= 200 :
                loss_mesh_smooth = mesh_bi_laplacian_smoothness_loss(vertices, faces_tensor)
                total_loss = total_loss + lambda_mesh_smooth * loss_mesh_smooth
                loss_dict['Mesh_Smooth'] = loss_mesh_smooth.item()

                print("laplace loss: ", loss_mesh_smooth)

            # 法线一致性损失
            if lambda_normal_consistency > 0:
                loss_normal_consistency = mesh_normal_consistency_loss(vertices, faces_tensor)
                total_loss = total_loss + lambda_normal_consistency * loss_normal_consistency
                loss_dict['Normal_Consistency'] = loss_normal_consistency.item()

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
            has_inf_grad = False
            for param in [fc_params['sdf'], fc_params['deform'], fc_params['weight']]:
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        has_nan_grad = True
                    if torch.isinf(param.grad).any():
                        has_inf_grad = True

                    if has_nan_grad and has_inf_grad:
                        break

            if has_nan_grad:
                print(f'[WARNING] NaN gradients at iteration {iteration}, skipping update...')

            if has_inf_grad:
                print(f'[WARNING] Inf gradients at iteration {iteration}, skipping update...')

            if has_nan_grad or has_inf_grad:
                exit()

            # 更新参数
            optimizer.step()

            # 裁剪 SDF 值
            with torch.no_grad():
                fc_params['sdf'].data = torch.clamp(fc_params['sdf'].data, -10.0, 10.0)

            if writer is not None:
                writer.add_scalar('Loss/Total', total_loss.item(), iteration)
                # 记录所有实际计算的 loss
                for loss_name, loss_value in loss_dict.items():
                    writer.add_scalar(f'Loss/{loss_name}', loss_value, iteration)

            # 更新进度条和日志
            if iteration % log_interval == 0 or iteration == num_iterations - 1:
                if writer is not None:
                    # 记录渲染图像
                    for i, (render_data, render_idx) in enumerate(zip(render_data_list, render_idx_list)):
                        if render_data.dim() == 3 and render_data.shape[-1] == 3:
                            render_data = render_data.permute(2, 0, 1)
                        writer.add_image(f'Render/Camera_{render_idx}', render_data, global_step=iteration)

            # 更新进度条（只显示实际使用的 loss）
            postfix_dict = {'loss': f'{total_loss.item():.4f}'}
            if 'Render' in loss_dict:
                postfix_dict['render'] = f'{loss_dict["Render"]:.4f}'
            if 'SDF_Smooth' in loss_dict:
                postfix_dict['sdf_sm'] = f'{loss_dict["SDF_Smooth"]:.6f}'
            if 'Dev' in loss_dict:
                postfix_dict['dev'] = f'{loss_dict["Dev"]:.6f}'
            if 'Normal_Consistency' in loss_dict:
                postfix_dict['normal'] = f'{loss_dict["Normal_Consistency"]:.6f}'
            pbar.set_postfix(postfix_dict)

        # 关闭 TensorBoard writer
        if writer is not None:
            writer.close()

        # 提取最终 mesh
        final_mesh, _, _ = FCConvertor.extractMesh(fc_params, training=False)

        return final_mesh
