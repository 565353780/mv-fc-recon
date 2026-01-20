import torch
import numpy as np
from typing import Callable, Optional, Tuple


def compute_sdf_gradients(
    sdf_func: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    eps: float = 1e-3,
    taps: int = 6,
    training: bool = False,
    sdf: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """使用有限差分计算 SDF 的梯度和 Hessian 对角元素

    参考: neural-angelo/neural_angelo/Model/neural_sdf.py

    Args:
        sdf_func: SDF 函数，输入 [..., 3] 点坐标，输出 [..., 1] SDF 值
        x: [..., 3] 查询点坐标
        eps: 有限差分步长
        taps: 采样点数量，支持 4 或 6
        training: 是否处于训练模式（训练时计算 Hessian）
        sdf: [..., 1] 预计算的 SDF 值（用于计算 Hessian，可选）

    Returns:
        gradient: [..., 3] SDF 梯度
        hessian: [..., 3] Hessian 对角元素（仅在 training=True 时返回）
    """
    if taps == 6:
        # 6-tap 中心差分法
        # 沿 x, y, z 三个方向分别计算正负偏移的 SDF 值
        eps_x = torch.tensor([eps, 0., 0.], dtype=x.dtype, device=x.device)  # [3]
        eps_y = torch.tensor([0., eps, 0.], dtype=x.dtype, device=x.device)  # [3]
        eps_z = torch.tensor([0., 0., eps], dtype=x.dtype, device=x.device)  # [3]

        sdf_x_pos = sdf_func(x + eps_x)  # [..., 1]
        sdf_x_neg = sdf_func(x - eps_x)  # [..., 1]
        sdf_y_pos = sdf_func(x + eps_y)  # [..., 1]
        sdf_y_neg = sdf_func(x - eps_y)  # [..., 1]
        sdf_z_pos = sdf_func(x + eps_z)  # [..., 1]
        sdf_z_neg = sdf_func(x - eps_z)  # [..., 1]

        # 一阶梯度（中心差分）
        gradient_x = (sdf_x_pos - sdf_x_neg) / (2 * eps)
        gradient_y = (sdf_y_pos - sdf_y_neg) / (2 * eps)
        gradient_z = (sdf_z_pos - sdf_z_neg) / (2 * eps)
        gradient = torch.cat([gradient_x, gradient_y, gradient_z], dim=-1)  # [..., 3]

        # 二阶梯度（Hessian 对角元素）
        if training:
            if sdf is None:
                sdf = sdf_func(x)  # [..., 1]
            hessian_xx = (sdf_x_pos + sdf_x_neg - 2 * sdf) / (eps ** 2)  # [..., 1]
            hessian_yy = (sdf_y_pos + sdf_y_neg - 2 * sdf) / (eps ** 2)  # [..., 1]
            hessian_zz = (sdf_z_pos + sdf_z_neg - 2 * sdf) / (eps ** 2)  # [..., 1]
            hessian = torch.cat([hessian_xx, hessian_yy, hessian_zz], dim=-1)  # [..., 3]
        else:
            hessian = None

    elif taps == 4:
        # 4-tap 四面体采样法
        eps_scaled = eps / np.sqrt(3)
        k1 = torch.tensor([1, -1, -1], dtype=x.dtype, device=x.device)  # [3]
        k2 = torch.tensor([-1, -1, 1], dtype=x.dtype, device=x.device)  # [3]
        k3 = torch.tensor([-1, 1, -1], dtype=x.dtype, device=x.device)  # [3]
        k4 = torch.tensor([1, 1, 1], dtype=x.dtype, device=x.device)  # [3]

        sdf1 = sdf_func(x + k1 * eps_scaled)  # [..., 1]
        sdf2 = sdf_func(x + k2 * eps_scaled)  # [..., 1]
        sdf3 = sdf_func(x + k3 * eps_scaled)  # [..., 1]
        sdf4 = sdf_func(x + k4 * eps_scaled)  # [..., 1]

        # 一阶梯度
        gradient = (k1 * sdf1 + k2 * sdf2 + k3 * sdf3 + k4 * sdf4) / (4.0 * eps_scaled)

        # 二阶梯度（Hessian 对角元素）
        if training:
            if sdf is None:
                sdf = sdf_func(x)  # [..., 1]
            # 4-tap 方法的结果是 trace，这里假设各分量相等
            hessian_xx = ((sdf1 + sdf2 + sdf3 + sdf4) / 2.0 - 2 * sdf) / (eps_scaled ** 2)  # [..., 1]
            hessian = torch.cat([hessian_xx, hessian_xx, hessian_xx], dim=-1) / 3.0  # [..., 3]
        else:
            hessian = None

    else:
        raise ValueError(f"Only support 4 or 6 taps, got {taps}")

    return gradient, hessian


def compute_sdf_gradients_from_grid(
    sdf_grid: torch.Tensor,
    x_nx3: torch.Tensor,
    query_points: torch.Tensor,
    resolution: int,
    eps: Optional[float] = None,
    training: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """从 SDF 网格计算查询点的梯度和 Hessian

    适用于 FlexiCubes 等基于网格的 SDF 表示。

    Args:
        sdf_grid: [N] SDF 网格值
        x_nx3: [N, 3] 网格顶点坐标
        query_points: [M, 3] 查询点坐标
        resolution: 网格分辨率
        eps: 有限差分步长，默认为网格间距
        training: 是否处于训练模式

    Returns:
        gradient: [M, 3] SDF 梯度
        hessian: [M, 3] Hessian 对角元素（仅在 training=True 时返回）
    """
    if eps is None:
        # 默认使用网格间距
        eps = 1.0 / resolution

    # 创建一个基于三线性插值的 SDF 函数
    def sdf_func(points):
        return trilinear_interpolate_sdf(sdf_grid, x_nx3, points, resolution)

    return compute_sdf_gradients(sdf_func, query_points, eps, taps=6, training=training)


def trilinear_interpolate_sdf(
    sdf_grid: torch.Tensor,
    x_nx3: torch.Tensor,
    query_points: torch.Tensor,
    resolution: int,
) -> torch.Tensor:
    """三线性插值查询 SDF 值

    Args:
        sdf_grid: [N] SDF 网格值，N = (resolution + 1)^3
        x_nx3: [N, 3] 网格顶点坐标，范围 [-1, 1]
        query_points: [..., 3] 查询点坐标，范围 [-1, 1]
        resolution: 网格分辨率

    Returns:
        sdf: [..., 1] 插值后的 SDF 值
    """
    original_shape = query_points.shape[:-1]
    points_flat = query_points.view(-1, 3)  # [M, 3]

    # 将坐标从 [-1, 1] 映射到 [0, resolution]
    # x_nx3 的范围是 [-1, 1]，网格点索引范围是 [0, resolution]
    grid_coords = (points_flat + 1.0) / 2.0 * resolution  # [M, 3]

    # 裁剪到有效范围
    grid_coords = torch.clamp(grid_coords, 0, resolution - 1e-6)

    # 获取整数索引和小数部分
    grid_coords_floor = grid_coords.floor().long()  # [M, 3]
    grid_coords_frac = grid_coords - grid_coords_floor.float()  # [M, 3]

    # 计算 8 个角点的索引
    fx, fy, fz = grid_coords_frac[:, 0:1], grid_coords_frac[:, 1:2], grid_coords_frac[:, 2:3]
    ix, iy, iz = grid_coords_floor[:, 0], grid_coords_floor[:, 1], grid_coords_floor[:, 2]

    # 确保索引不越界
    ix1 = torch.clamp(ix + 1, 0, resolution)
    iy1 = torch.clamp(iy + 1, 0, resolution)
    iz1 = torch.clamp(iz + 1, 0, resolution)

    # 计算线性索引（假设网格按 x, y, z 顺序排列）
    stride_y = resolution + 1
    stride_z = (resolution + 1) ** 2

    def get_idx(x, y, z):
        return x + y * stride_y + z * stride_z

    # 获取 8 个角点的 SDF 值
    c000 = sdf_grid[get_idx(ix, iy, iz)]
    c001 = sdf_grid[get_idx(ix, iy, iz1)]
    c010 = sdf_grid[get_idx(ix, iy1, iz)]
    c011 = sdf_grid[get_idx(ix, iy1, iz1)]
    c100 = sdf_grid[get_idx(ix1, iy, iz)]
    c101 = sdf_grid[get_idx(ix1, iy, iz1)]
    c110 = sdf_grid[get_idx(ix1, iy1, iz)]
    c111 = sdf_grid[get_idx(ix1, iy1, iz1)]

    # 三线性插值
    c00 = c000 * (1 - fz.squeeze(-1)) + c001 * fz.squeeze(-1)
    c01 = c010 * (1 - fz.squeeze(-1)) + c011 * fz.squeeze(-1)
    c10 = c100 * (1 - fz.squeeze(-1)) + c101 * fz.squeeze(-1)
    c11 = c110 * (1 - fz.squeeze(-1)) + c111 * fz.squeeze(-1)

    c0 = c00 * (1 - fy.squeeze(-1)) + c01 * fy.squeeze(-1)
    c1 = c10 * (1 - fy.squeeze(-1)) + c11 * fy.squeeze(-1)

    sdf = c0 * (1 - fx.squeeze(-1)) + c1 * fx.squeeze(-1)

    # 恢复原始形状
    sdf = sdf.view(*original_shape, 1)
    return sdf
