import torch


def curvature_loss(hessian, outside=None):
    """计算曲率 loss（基于 Laplacian）

    Laplacian 是 Hessian 矩阵的迹（对角元素之和），
    该 loss 惩罚 SDF 的曲率，用于平滑正则化。

    参考: neural-angelo/neural_angelo/Loss/curvature.py

    Args:
        hessian: [..., 3] Hessian 对角元素，支持 [B, R, N, 3] 或 [M, 3] 格式
        outside: 场景外标志（可选），True 表示在场景外

    Returns:
        loss: curvature loss 标量 (tensor)
    """
    if hessian is None:
        return torch.tensor(0.0)

    # Laplacian = trace(Hessian) = hessian_xx + hessian_yy + hessian_zz
    laplacian = hessian.sum(dim=-1).abs()  # [...]
    laplacian = laplacian.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    if outside is not None:
        # 处理不同维度的 outside 标志
        if outside.dim() < laplacian.dim():
            # [B, R, 1] -> [B, R, N] 通过 expand
            outside = outside.expand_as(laplacian)
        elif outside.dim() > laplacian.dim():
            # [B, R, 1] 与 [M] 格式不兼容，忽略 outside
            return laplacian.mean()
        # 只在场景内部计算 loss
        inside_mask = (~outside).float()
        num_inside = inside_mask.sum()
        if num_inside > 0:
            return (laplacian * inside_mask).sum() / num_inside
        else:
            return laplacian.mean()
    else:
        return laplacian.mean()


def get_curvature_weight(
    current_iteration: int,
    init_weight: float,
    warm_up_end: int,
    growth_rate: float = 1.0,
    anneal_levels: int = 1,
) -> float:
    """计算动态调整的曲率 loss 权重

    参考: neural-angelo/neural_angelo/Module/trainer.py 中的 get_curvature_weight

    在 warm-up 阶段，权重从 0 线性增长到 init_weight；
    在 warm-up 之后，权重随 coarse-to-fine 级别衰减。

    Args:
        current_iteration: 当前迭代次数
        init_weight: 初始权重
        warm_up_end: warm-up 结束的迭代次数
        growth_rate: coarse-to-fine 的增长率
        anneal_levels: 当前激活的 coarse-to-fine 级别

    Returns:
        weight: 当前的曲率 loss 权重
    """
    if current_iteration <= warm_up_end:
        # Warm-up 阶段：线性增长
        return current_iteration / warm_up_end * init_weight
    else:
        # Warm-up 之后：根据 coarse-to-fine 级别衰减
        decay_factor = growth_rate ** (anneal_levels - 1)
        return init_weight / decay_factor
