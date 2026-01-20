import torch


def eikonal_loss(gradients, outside=None):
    """计算 eikonal loss

    Eikonal 方程要求 SDF 的梯度模长为 1，即 ||∇SDF|| = 1。
    该 loss 惩罚梯度模长偏离 1 的程度。

    参考: neural-angelo/neural_angelo/Loss/eikonal.py

    Args:
        gradients: [..., 3] 梯度值，支持 [B, R, N, 3] 或 [M, 3] 格式
        outside: 场景外标志（可选），True 表示在场景外

    Returns:
        loss: eikonal loss 标量
    """
    if gradients is None:
        return torch.tensor(0.0)

    gradient_error = (gradients.norm(dim=-1) - 1.0) ** 2  # [...]
    gradient_error = gradient_error.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)

    if outside is not None:
        # 处理不同维度的 outside 标志
        if outside.dim() < gradient_error.dim():
            # [B, R, 1] -> [B, R, N] 通过 expand
            outside = outside.expand_as(gradient_error)
        elif outside.dim() > gradient_error.dim():
            # [B, R, 1] 与 [M] 格式不兼容，忽略 outside
            return gradient_error.mean()
        # 只在场景内部计算 loss
        inside_mask = (~outside).float()
        num_inside = inside_mask.sum()
        if num_inside > 0:
            return (gradient_error * inside_mask).sum() / num_inside
        else:
            return gradient_error.mean()
    else:
        return gradient_error.mean()
