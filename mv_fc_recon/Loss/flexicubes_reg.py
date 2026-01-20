"""
FlexiCubes 专用正则化损失函数

FlexiCubes 的 SDF 表示与传统 SDF（如 NeuS、NeuralAngelo）有本质区别：
1. FlexiCubes 的 SDF 只需要符号正确（正=外部，负=内部），不需要是真正的距离场
2. 强制 ||∇SDF|| = 1 的 Eikonal 约束对 FlexiCubes 是错误的
3. 基于 Hessian 的曲率约束在离散网格上会放大噪声

本文件提供适合 FlexiCubes 的正则化方法。
"""

import torch
from typing import Dict, Optional


def sdf_smoothness_loss(
    sdf: torch.Tensor,
    grid_edges: torch.Tensor,
) -> torch.Tensor:
    """SDF 平滑正则化：惩罚相邻网格点 SDF 差异过大

    这比 Eikonal loss 更适合 FlexiCubes，因为它：
    1. 不强制梯度模长为 1
    2. 只惩罚局部不平滑，允许 SDF 有任意尺度

    Args:
        sdf: [N] SDF 网格值
        grid_edges: [E, 2] 边索引

    Returns:
        loss: 平滑损失
    """
    # 获取边两端的 SDF 值
    sdf_edge = sdf[grid_edges]  # [E, 2]
    # 计算差异的平方（L2 平滑）
    diff = (sdf_edge[:, 0] - sdf_edge[:, 1]) ** 2
    return diff.mean()


def sdf_gradient_smoothness_loss(
    sdf: torch.Tensor,
    grid_edges: torch.Tensor,
    x_nx3: torch.Tensor,
) -> torch.Tensor:
    """SDF 梯度平滑正则化：惩罚梯度变化过大

    与 Eikonal 不同，这里不强制梯度模长为 1，
    而是惩罚梯度在空间上的变化（二阶平滑）。

    Args:
        sdf: [N] SDF 网格值
        grid_edges: [E, 2] 边索引
        x_nx3: [N, 3] 网格顶点坐标

    Returns:
        loss: 梯度平滑损失
    """
    # 计算每条边上的 SDF 梯度（近似）
    sdf_edge = sdf[grid_edges]  # [E, 2]
    pos_edge = x_nx3[grid_edges]  # [E, 2, 3]

    # 边方向和长度
    edge_vec = pos_edge[:, 1] - pos_edge[:, 0]  # [E, 3]
    edge_len = edge_vec.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [E, 1]

    # SDF 沿边方向的梯度
    grad_along_edge = (sdf_edge[:, 1] - sdf_edge[:, 0]).unsqueeze(-1) / edge_len  # [E, 1]

    # 惩罚梯度的变化（方差）
    grad_mean = grad_along_edge.mean()
    grad_var = ((grad_along_edge - grad_mean) ** 2).mean()

    return grad_var


def sdf_sign_consistency_loss(
    sdf: torch.Tensor,
    grid_edges: torch.Tensor,
    margin: float = 0.01,
) -> torch.Tensor:
    """SDF 符号一致性损失：软化版的 SDF 正则化

    原始的 BCE 版本在 SDF 接近 0 时梯度爆炸。
    这个版本使用更稳定的 margin-based loss。

    对于符号变化的边（表面穿过的边），鼓励两端 SDF 值接近 0。
    对于符号相同的边，不施加约束。

    Args:
        sdf: [N] SDF 网格值
        grid_edges: [E, 2] 边索引
        margin: SDF 值的 margin

    Returns:
        loss: 符号一致性损失
    """
    sdf_edge = sdf[grid_edges]  # [E, 2]

    # 找到符号变化的边（表面穿过的边）
    sign_change = (sdf_edge[:, 0] * sdf_edge[:, 1]) < 0  # [E]

    if sign_change.sum() == 0:
        return torch.tensor(0.0, device=sdf.device)

    # 对于符号变化的边，鼓励两端 SDF 绝对值较小
    # 这样表面位置更精确
    sdf_abs = sdf_edge[sign_change].abs()  # [E', 2]

    # 使用 soft margin loss：max(|sdf| - margin, 0)^2
    # 这比 BCE 更稳定
    excess = torch.relu(sdf_abs - margin)
    loss = (excess ** 2).mean()

    return loss


def weight_regularization_loss(
    weight: torch.Tensor,
    target_scale: float = 0.5,
) -> torch.Tensor:
    """FlexiCubes 权重正则化

    约束 alpha, beta, gamma 权重不要偏离太远，
    防止某些立方体使用极端权重导致表面扭曲。

    Args:
        weight: [F, 21] FlexiCubes 权重
            - [:, :12]: beta (12 条边的插值权重)
            - [:, 12:20]: alpha (8 个顶点的权重)
            - [:, 20]: gamma_f (每个立方体的权重)
        target_scale: 目标缩放因子

    Returns:
        loss: 权重正则化损失
    """
    # 分离不同类型的权重
    beta = weight[:, :12]  # [F, 12]
    alpha = weight[:, 12:20]  # [F, 8]
    gamma = weight[:, 20]  # [F]

    # L2 正则化，鼓励权重接近 0（FlexiCubes 默认行为）
    loss_beta = (beta ** 2).mean()
    loss_alpha = (alpha ** 2).mean()
    loss_gamma = (gamma ** 2).mean()

    return loss_beta + loss_alpha + loss_gamma


def mesh_laplacian_smoothness_loss(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """网格 Laplacian 平滑损失

    直接在提取的网格顶点上计算 Laplacian 平滑，
    这比在 SDF 上计算曲率更直接有效。

    Args:
        vertices: [V, 3] 网格顶点
        faces: [F, 3] 网格面片

    Returns:
        loss: Laplacian 平滑损失
    """
    device = vertices.device
    num_vertices = vertices.shape[0]

    # 构建邻接关系
    # 从面片中提取边
    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ], dim=0)  # [3F, 2]

    # 去重并创建双向边
    edges = torch.cat([edges, edges.flip(1)], dim=0)  # [6F, 2]
    edges = torch.unique(edges, dim=0)  # [E', 2]

    # 计算每个顶点的邻居均值
    # 使用 scatter_add 来累加邻居位置
    neighbor_sum = torch.zeros_like(vertices)  # [V, 3]
    neighbor_count = torch.zeros(num_vertices, device=device)  # [V]

    src_idx = edges[:, 0]  # [E']
    dst_idx = edges[:, 1]  # [E']

    # 累加邻居位置
    neighbor_sum.scatter_add_(0, dst_idx.unsqueeze(1).expand(-1, 3), vertices[src_idx])
    neighbor_count.scatter_add_(0, dst_idx, torch.ones(len(dst_idx), device=device))

    # 避免除以 0
    neighbor_count = neighbor_count.clamp(min=1)

    # 邻居均值
    neighbor_mean = neighbor_sum / neighbor_count.unsqueeze(1)  # [V, 3]

    # Laplacian = 顶点位置 - 邻居均值
    laplacian = vertices - neighbor_mean  # [V, 3]

    # L2 损失
    loss = (laplacian ** 2).sum(dim=-1).mean()

    return loss


def mesh_normal_consistency_loss(
    vertices: torch.Tensor,
    faces: torch.Tensor,
) -> torch.Tensor:
    """网格法线一致性损失

    惩罚相邻面片法线差异过大，用于平滑表面。

    Args:
        vertices: [V, 3] 网格顶点
        faces: [F, 3] 网格面片

    Returns:
        loss: 法线一致性损失
    """
    # 计算面片法线
    v0 = vertices[faces[:, 0]]  # [F, 3]
    v1 = vertices[faces[:, 1]]  # [F, 3]
    v2 = vertices[faces[:, 2]]  # [F, 3]

    e1 = v1 - v0  # [F, 3]
    e2 = v2 - v0  # [F, 3]

    face_normals = torch.cross(e1, e2, dim=-1)  # [F, 3]
    face_normals = face_normals / (face_normals.norm(dim=-1, keepdim=True).clamp(min=1e-8))

    # 构建面片邻接关系（共享边的面片）
    # 简化版：使用共享顶点的面片
    num_faces = faces.shape[0]
    device = vertices.device

    # 从边构建面片邻接
    edges = torch.cat([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ], dim=0)  # [3F, 2]

    # 排序边的顶点索引，使得 (a, b) 和 (b, a) 相同
    edges_sorted = torch.sort(edges, dim=1)[0]  # [3F, 2]

    # 为每条边记录其所属的面片索引
    face_indices = torch.arange(num_faces, device=device).repeat(3)  # [3F]

    # 找到共享边的面片对
    # 使用字典来找到共享边
    edge_to_faces = {}
    for i, (e, f) in enumerate(zip(edges_sorted.tolist(), face_indices.tolist())):
        key = tuple(e)
        if key not in edge_to_faces:
            edge_to_faces[key] = []
        edge_to_faces[key].append(f)

    # 收集相邻面片对
    adjacent_pairs = []
    for faces_list in edge_to_faces.values():
        if len(faces_list) == 2:
            adjacent_pairs.append(faces_list)

    if len(adjacent_pairs) == 0:
        return torch.tensor(0.0, device=device)

    adjacent_pairs = torch.tensor(adjacent_pairs, device=device, dtype=torch.long)  # [P, 2]

    # 计算相邻面片法线的点积
    n1 = face_normals[adjacent_pairs[:, 0]]  # [P, 3]
    n2 = face_normals[adjacent_pairs[:, 1]]  # [P, 3]

    # 1 - dot product = 0 表示法线相同，= 2 表示法线相反
    dot_product = (n1 * n2).sum(dim=-1)  # [P]
    loss = (1 - dot_product).mean()

    return loss


def compute_flexicubes_regularization(
    fc_params: Dict,
    vertices: Optional[torch.Tensor] = None,
    faces: Optional[torch.Tensor] = None,
    lambda_sdf_smooth: float = 0.1,
    lambda_weight_reg: float = 0.01,
    lambda_mesh_smooth: float = 0.0,
    lambda_normal_consistency: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """计算 FlexiCubes 专用的正则化损失

    Args:
        fc_params: FlexiCubes 参数字典
        vertices: [V, 3] 网格顶点（可选，用于网格平滑）
        faces: [F, 3] 网格面片（可选，用于网格平滑）
        lambda_sdf_smooth: SDF 平滑权重
        lambda_weight_reg: 权重正则化权重
        lambda_mesh_smooth: 网格 Laplacian 平滑权重
        lambda_normal_consistency: 法线一致性权重

    Returns:
        losses: 包含各项损失的字典
    """
    device = fc_params['sdf'].device
    losses = {}

    # SDF 平滑损失
    if lambda_sdf_smooth > 0:
        losses['sdf_smooth'] = lambda_sdf_smooth * sdf_smoothness_loss(
            fc_params['sdf'],
            fc_params['grid_edges'],
        )
    else:
        losses['sdf_smooth'] = torch.tensor(0.0, device=device)

    # 权重正则化损失
    if lambda_weight_reg > 0:
        losses['weight_reg'] = lambda_weight_reg * weight_regularization_loss(
            fc_params['weight'],
        )
    else:
        losses['weight_reg'] = torch.tensor(0.0, device=device)

    # 网格 Laplacian 平滑损失
    if lambda_mesh_smooth > 0 and vertices is not None and faces is not None:
        losses['mesh_smooth'] = lambda_mesh_smooth * mesh_laplacian_smoothness_loss(
            vertices, faces,
        )
    else:
        losses['mesh_smooth'] = torch.tensor(0.0, device=device)

    # 法线一致性损失
    if lambda_normal_consistency > 0 and vertices is not None and faces is not None:
        losses['normal_consistency'] = lambda_normal_consistency * mesh_normal_consistency_loss(
            vertices, faces,
        )
    else:
        losses['normal_consistency'] = torch.tensor(0.0, device=device)

    return losses
