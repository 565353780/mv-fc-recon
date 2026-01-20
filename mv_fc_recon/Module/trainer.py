import os
import cv2
import torch
import trimesh
import numpy as np
from tqdm import tqdm
from typing import Union, List, Dict
from torch.utils.tensorboard import SummaryWriter

from camera_control.Module.rgbd_camera import RGBDCamera
from camera_control.Module.nvdiffrast_renderer import NVDiffRastRenderer

from mv_fc_recon.Loss.sdf_reg import computeSDFRegLoss
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
        batch_size: int = 8,
        lr: float = 0.01,
        lambda_dev: float = 0.5,
        lambda_sdf_reg: float = 0.2,
        log_interval: int = 1,
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
            batch_size: 每次迭代采样的视角数量
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
                curr_mesh.export(log_dir + 'start_fc_mesh.ply')

        # 训练循环
        pbar = tqdm(range(num_iterations), desc='FlexiCubes Optimization')
        for iteration in pbar:
            optimizer.zero_grad()

            # 从FlexiCubes参数提取mesh（只提取一次，用于所有视角）
            current_mesh, vertices, L_dev = FCConvertor.extractMesh(
                fc_params, training=True
            )

            # 随机采样batch_size个视角
            num_cameras = len(camera_list)
            if batch_size >= num_cameras:
                batch_indices = list(range(num_cameras))
            else:
                batch_indices = np.random.choice(num_cameras, batch_size, replace=False).tolist()

            total_render_loss = 0.0
            first_render_rgb = None  # 保存第一个视角的渲染结果用于TensorBoard

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
                rasterize_output = render_dict['rasterize_output']  # [H, W, 4]

                # 确保渲染结果在[0, 1]范围
                if render_rgb.max() > 1.0:
                    render_rgb = render_rgb / 255.0

                # 保存第一个视角的渲染结果用于TensorBoard
                if first_render_rgb is None:
                    first_render_rgb = render_rgb.clone()

                # 计算mask损失（silhouette loss）
                # 渲染的mask
                render_mask = (rasterize_output[..., 3:4] > 0).float()  # [H, W, 1]
                # 目标mask（假设背景是白色）
                bg_threshold = 0.95
                target_mask = (target_rgb.mean(dim=-1, keepdim=True) < bg_threshold).float()  # [H, W, 1]

                # Mask loss (IoU-based)
                intersection = (render_mask * target_mask).sum()
                union = render_mask.sum() + target_mask.sum() - intersection
                mask_loss = 1.0 - intersection / (union + 1e-8)

                # RGB loss（只在有效区域计算）
                valid_mask = render_mask * target_mask
                rgb_loss = ((render_rgb - target_rgb).abs() * valid_mask).sum() / (valid_mask.sum() * 3 + 1e-8)

                # 组合渲染损失
                render_loss = mask_loss + rgb_loss
                total_render_loss += render_loss

            # 平均渲染损失
            avg_render_loss = total_render_loss / len(batch_indices)

            # FlexiCubes developability正则化损失
            loss_dev = L_dev.mean() if L_dev is not None and L_dev.numel() > 0 else torch.tensor(0.0, device=device)

            # SDF边缘正则化损失
            loss_sdf_reg = computeSDFRegLoss(fc_params['sdf'], grid_edges)

            # 总损失
            total_loss = avg_render_loss + lambda_dev * loss_dev + lambda_sdf_reg * loss_sdf_reg

            # 反向传播
            total_loss.backward()

            # 更新参数
            optimizer.step()

            # 记录到TensorBoard
            if writer is not None:
                # 记录所有loss
                writer.add_scalar('Loss/Total', total_loss.item(), iteration)
                writer.add_scalar('Loss/Render', avg_render_loss.item(), iteration)
                writer.add_scalar('Loss/Dev', loss_dev.item(), iteration)
                writer.add_scalar('Loss/SDF_Reg', loss_sdf_reg.item(), iteration)

                # 记录渲染的RGB图像（使用第一个batch的渲染结果）
                if first_render_rgb is not None and len(batch_indices) > 0:
                    first_idx = batch_indices[0]
                    render_rgb = first_render_rgb.clone()
                    if render_rgb.dim() == 3 and render_rgb.shape[-1] == 3:
                        render_rgb = render_rgb.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
                    writer.add_image(f'Render/Camera_{first_idx}', render_rgb, global_step=iteration)

            # 更新进度条
            if iteration % log_interval == 0 or iteration == num_iterations - 1:
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

            # 从FlexiCubes参数提取mesh
            current_mesh, vertices, L_dev = FCConvertor.extractMesh(
                fc_params, training=True
            )

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

            # 反向传播
            total_loss.backward()

            # 更新参数
            optimizer.step()

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
