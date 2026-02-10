import os
import torch
import trimesh
import warp as wp
import numpy as np
from typing import Union, Dict, Optional, Tuple
from kaolin.ops.conversions import FlexiCubes

from mv_fc_recon.Method.io import loadMeshFile


# 1. 初始化 Warp
wp.init()

@wp.kernel
def compute_sdf_kernel(
    mesh: wp.uint64,                 # Mesh 句柄
    query_points: wp.array(dtype=wp.vec3),
    out_sdf: wp.array(dtype=float),
    out_gradients: wp.array(dtype=wp.vec3)  # 可选：SDF 梯度（即方向）
):
    tid = wp.tid()
    p = query_points[tid]

    # max_dist 设置为一个足够大的数
    # MeshQueryPoint 包含: result, face, u, v, sign
    query_res = wp.mesh_query_point(mesh, p, 1.0e6)

    if query_res.result:
        # 使用重心坐标计算最近点位置
        face_idx = query_res.face
        u = query_res.u
        v = query_res.v
        closest_p = wp.mesh_eval_position(mesh, face_idx, u, v)

        # 计算距离 (Unsigned)
        dist = wp.length(p - closest_p)

        # 使用 query_res.sign 直接获取符号（正数表示外部，负数表示内部）
        sdf_val = query_res.sign * dist

        out_sdf[tid] = sdf_val

        # 计算方向向量
        diff = p - closest_p

        # 计算梯度（SDF 的导数就是指向表面的单位向量）
        if dist > 1e-6:
            out_gradients[tid] = wp.normalize(diff)
        else:
            # 在表面上直接使用法线
            normal = wp.mesh_eval_face_normal(mesh, face_idx)
            out_gradients[tid] = normal

    else:
        # 如果超出 max_dist
        out_sdf[tid] = 1.0e6
        out_gradients[tid] = wp.vec3(0.0, 0.0, 0.0)

def compute_gt_sdf_batch(wp_mesh, points: torch.Tensor) -> torch.Tensor:
    """使用 Warp 计算 GT SDF。

    Args:
        points: [N, 3] 或 [..., 3] 采样点坐标。

    Returns:
        sdf: [N] 或 [...] GT SDF 值。
    """
    original_shape = points.shape[:-1]
    points_flat = points.view(-1, 3)
    num_points = points_flat.shape[0]

    # 准备查询点
    points_np = points_flat.detach().cpu().numpy().astype(np.float32)

    # 创建 Warp 数组
    wp_points = wp.array(points_np, dtype=wp.vec3)
    wp_sdf = wp.zeros(num_points, dtype=float)
    wp_gradients = wp.zeros(num_points, dtype=wp.vec3)

    # 运行 kernel
    wp.launch(
        kernel=compute_sdf_kernel,
        dim=num_points,
        inputs=[wp_mesh.id, wp_points, wp_sdf, wp_gradients]
    )

    # 同步并转换回 PyTorch
    wp.synchronize()
    sdf_np = wp_sdf.numpy()
    sdf = torch.from_numpy(sdf_np).float().to(points.device)

    # 恢复原始形状
    sdf = sdf.view(*original_shape)
    return sdf

class FCConvertor(object):
    def __init__(self) -> None:
        return

    @staticmethod
    def meshToSDF(
        mesh: trimesh.Trimesh,
        query_points: torch.Tensor,
    ) -> torch.Tensor:
        """
        在指定查询点计算mesh的SDF值（使用trimesh的精确计算）

        Args:
            mesh: trimesh.Trimesh对象
            query_points: [N, 3] 查询点坐标（可以是任意坐标范围）
            device: 计算设备

        Returns:
            sdf: [N] SDF值（内部为负，外部为正）
        """
        vertices = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.int32).flatten()

        wp_mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3),
            indices=wp.array(faces, dtype=int)
        )

        return compute_gt_sdf_batch(wp_mesh, query_points)

    @staticmethod
    def createFC(
        mesh: Union[str, trimesh.Trimesh, None] = None,
        resolution: int = 64,
        device: str = 'cuda:0',
    ) -> Optional[Dict]:
        """
        从三角网格创建FlexiCubes参数

        参考: https://github.com/nv-tlabs/FlexiCubes/blob/main/examples/optimization.ipynb

        Args:
            mesh: 网格文件路径或trimesh对象，如果为None则随机初始化SDF
            resolution: FlexiCubes分辨率（体素网格分辨率）
            device: 计算设备

        Returns:
            dict包含: fc, sdf, deform, weight, x_nx3, cube_fx8, grid_edges
        """
        if isinstance(mesh, str):
            if not os.path.exists(mesh):
                print('[ERROR][FCConvertor::createFC]')
                print('\t mesh file not exist!')
                print('\t mesh_file_path:', mesh)
                return None
            mesh = loadMeshFile(mesh)
            # mesh.vertices =  mesh.vertices * 0.5 
            # print("debug: ", mesh.vertices.shape, mesh.faces.shape )

        # 创建FlexiCubes对象
        fc = FlexiCubes(device=device)

        # 构建体素网格
        x_nx3, cube_fx8 = fc.construct_voxel_grid(resolution)
        # x_nx3: [N, 3] 网格顶点坐标，范围[-1, 1]
        # cube_fx8: [F, 8] 每个立方体的8个顶点索引

        if mesh is not None:
            sdf_values = FCConvertor.meshToSDF(mesh, x_nx3)
        else:
            # 随机初始化SDF（参考官方示例）
            sdf_values = torch.rand_like(x_nx3[:, 0]) - 0.1

        # 创建可学习参数
        sdf = torch.nn.Parameter(sdf_values.clone().detach(), requires_grad=True)

        # deform: 顶点位移，形状与x_nx3相同 [N, 3]
        deform = torch.nn.Parameter(torch.zeros_like(x_nx3), requires_grad=True)

        # weight: FlexiCubes特有的权重
        # 参考官方示例：weight形状为 [F, 21]
        # beta: [:, :12] - 12个边的插值权重
        # alpha: [:, 12:20] - 8个顶点的权重
        # gamma_f: [:, 20] - 每个立方体的权重
        num_cubes = cube_fx8.shape[0]
        weight = torch.nn.Parameter(
            torch.zeros((num_cubes, 21), dtype=torch.float32, device=device),
            requires_grad=True
        )

        # 获取所有边用于正则化损失（参考官方示例）
        all_edges = cube_fx8[:, fc.cube_edges].reshape(-1, 2)
        grid_edges = torch.unique(all_edges, dim=0)

        return {
            'fc': fc,
            'sdf': sdf,
            'deform': deform,
            'weight': weight,
            'x_nx3': x_nx3,
            'cube_fx8': cube_fx8,
            'resolution': resolution,
            'grid_edges': grid_edges,
        }

    @staticmethod
    def extractMesh(
        fc_params: Dict,
        training: bool = True,
    ) -> Tuple[trimesh.Trimesh, torch.Tensor, torch.Tensor]:
        """
        从FlexiCubes参数提取三角网格

        参考: https://github.com/nv-tlabs/FlexiCubes/blob/main/examples/optimization.ipynb

        Args:
            fc_params: createFC返回的参数字典
            training: 是否处于训练模式（影响L_dev的计算）

        Returns:
            tuple: (mesh, vertices, faces, L_dev)
                - mesh: trimesh.Trimesh对象
                - vertices: [V, 3] 顶点tensor（保持梯度）
                - L_dev: developability损失
        """
        fc = fc_params['fc']
        sdf = fc_params['sdf']
        deform = fc_params['deform']
        weight = fc_params['weight']
        x_nx3 = fc_params['x_nx3']
        cube_fx8 = fc_params['cube_fx8']
        resolution = fc_params['resolution']

        # 检查输入参数是否包含NaN或Inf，创建清理后的副本用于提取
        sdf_clean = sdf.clone()
        if torch.isnan(sdf_clean).any() or torch.isinf(sdf_clean).any():
            # 如果SDF包含NaN或Inf，使用裁剪后的值
            sdf_clean = torch.clamp(sdf_clean, -10.0, 10.0)
            sdf_clean = torch.where(torch.isnan(sdf_clean), torch.zeros_like(sdf_clean), sdf_clean)
            sdf_clean = torch.where(torch.isinf(sdf_clean), torch.zeros_like(sdf_clean), sdf_clean)
        else:
            # 即使没有NaN/Inf，也进行裁剪以防止极端值
            sdf_clean = torch.clamp(sdf_clean, -10.0, 10.0)
        
        deform_clean = deform.clone()
        if torch.isnan(deform_clean).any() or torch.isinf(deform_clean).any():
            # 如果deform包含NaN或Inf，设置为0
            deform_clean = torch.where(torch.isnan(deform_clean), torch.zeros_like(deform_clean), deform_clean)
            deform_clean = torch.where(torch.isinf(deform_clean), torch.zeros_like(deform_clean), deform_clean)
        
        weight_clean = weight.clone()
        if torch.isnan(weight_clean).any() or torch.isinf(weight_clean).any():
            # 如果weight包含NaN或Inf，设置为0
            weight_clean = torch.where(torch.isnan(weight_clean), torch.zeros_like(weight_clean), weight_clean)
            weight_clean = torch.where(torch.isinf(weight_clean), torch.zeros_like(weight_clean), weight_clean)

        # 应用变形（参考官方示例：使用tanh限制变形范围）
        # 变形范围限制为网格单元大小的一半
        max_deform = (1.0 - 1e-8) / (resolution * 2)
        grid_verts = x_nx3 + max_deform * torch.tanh(deform_clean)

        # 使用FlexiCubes提取mesh
        # 参考官方API：
        # beta: [F, 12] - 12条边的插值权重
        # alpha: [F, 8] - 8个顶点的权重
        # gamma_f: [F] - 每个立方体的权重
        try:
            vertices, faces, L_dev = fc(
                grid_verts,           # voxelgrid_vertices
                sdf_clean,            # scalar_field (使用清理后的SDF)
                cube_fx8,             # cube_idx
                resolution,           # resolution
                beta=weight_clean[:, :12],        # 12条边的插值权重
                alpha=weight_clean[:, 12:20],     # 8个顶点的权重
                gamma_f=weight_clean[:, 20],      # 每个立方体的权重（注意是1D）
                training=training,
            )

            # 检查提取的顶点和面片是否有效
            if vertices is None or faces is None:
                raise ValueError("FlexiCubes returned None vertices or faces")
            
            if len(vertices) == 0 or len(faces) == 0:
                raise ValueError("FlexiCubes returned empty mesh")

            # 检查是否包含NaN或Inf
            if torch.isnan(vertices).any() or torch.isinf(vertices).any():
                raise ValueError("Extracted vertices contain NaN or Inf")

            mesh = trimesh.Trimesh(
                vertices=vertices.detach().cpu().numpy(),
                faces=faces.cpu().numpy(),
            )

            # 尝试修复网格（如果可能）
            try:
                # 检查并修复网格
                if hasattr(mesh, 'fill_holes'):
                    mesh.fill_holes()
                if hasattr(mesh, 'remove_duplicate_faces'):
                    mesh.remove_duplicate_faces()
                if hasattr(mesh, 'remove_unreferenced_vertices'):
                    mesh.remove_unreferenced_vertices()
            except Exception:
                # 如果修复失败，继续使用原始网格
                pass

        except Exception as e:
            # 如果提取失败，返回一个空的网格
            print(f'[WARNING][FCConvertor::extractMesh] Failed to extract mesh: {e}')
            # 创建一个最小的空网格（使用正确的形状）
            device = sdf.device
            empty_vertices = torch.zeros((0, 3), device=device, dtype=torch.float32)
            # 创建空网格时使用正确的numpy数组形状
            empty_vertices_np = np.zeros((0, 3), dtype=np.float32)
            empty_faces_np = np.zeros((0, 3), dtype=np.int32)
            empty_mesh = trimesh.Trimesh(vertices=empty_vertices_np, faces=empty_faces_np)
            empty_L_dev = torch.tensor(0.0, device=device, dtype=torch.float32)
            return empty_mesh, empty_vertices, empty_L_dev

        return mesh, vertices, L_dev
