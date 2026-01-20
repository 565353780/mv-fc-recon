import torch


def computeSDFRegLoss(
    sdf: torch.Tensor,
    grid_edges: torch.Tensor,
) -> torch.Tensor:
    """
    计算SDF正则化损失（边缘平滑约束）

    参考: https://github.com/nv-tlabs/FlexiCubes

    Args:
        sdf: [N] SDF值
        grid_edges: [E, 2] 边索引

    Returns:
        reg_loss: 正则化损失
    """
    sdf_f1x6x2 = sdf[grid_edges.reshape(-1)].reshape(-1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0],
        (sdf_f1x6x2[..., 1] > 0).float()
    ) + torch.nn.functional.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 1],
        (sdf_f1x6x2[..., 0] > 0).float()
    )
    return sdf_diff
