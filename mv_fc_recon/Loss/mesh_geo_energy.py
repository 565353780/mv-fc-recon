import torch

# 数值稳定常数：避免除零、梯度爆炸与 acos/atan2 奇异点
_EPS_AREA = 1e-6
_EPS_DENOM = 1e-8
_EPS_ACOS = 1e-7   # acos 输入远离 ±1，避免 d(acos)/dx 在边界处为 inf
_COT_MIN = -1e3
_COT_MAX = 1e3


def _corner_angles_safe(v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> tuple:
    """
    每个三角面三个顶角（弧度）。分母加 epsilon 避免除零；
    cos 限制在 (-1+eps, 1-eps) 避免 acos(±1) 处梯度爆炸。
    """
    e01 = v1 - v0
    e02 = v2 - v0
    e12 = v2 - v1
    e10 = v0 - v1
    e20 = v0 - v2
    e21 = v1 - v2
    n01 = torch.norm(e01, dim=1, keepdim=False) + _EPS_DENOM
    n02 = torch.norm(e02, dim=1, keepdim=False) + _EPS_DENOM
    n12 = torch.norm(e12, dim=1, keepdim=False) + _EPS_DENOM
    n10 = torch.norm(e10, dim=1, keepdim=False) + _EPS_DENOM
    n20 = torch.norm(e20, dim=1, keepdim=False) + _EPS_DENOM
    n21 = torch.norm(e21, dim=1, keepdim=False) + _EPS_DENOM
    cos0 = (e01 * e02).sum(dim=1) / (n01 * n02)
    cos1 = (e12 * (-e10)).sum(dim=1) / (n12 * n10)
    cos2 = (e20 * e21).sum(dim=1) / (n20 * n21)
    cos0 = cos0.clamp(-1.0 + _EPS_ACOS, 1.0 - _EPS_ACOS)
    cos1 = cos1.clamp(-1.0 + _EPS_ACOS, 1.0 - _EPS_ACOS)
    cos2 = cos2.clamp(-1.0 + _EPS_ACOS, 1.0 - _EPS_ACOS)
    return torch.acos(cos0), torch.acos(cos1), torch.acos(cos2)


def thin_plate_energy(
    V: torch.Tensor,
    F: torch.Tensor,
    with_gauss: bool = True,
):
    """
    离散 thin-plate 能量（矩阵自由、梯度完整、数值稳定）:
        E = ||M L V||^2 - 2 * sum_i (2*pi - sum_j theta_j)

    V: [n, 3]，需 requires_grad=True
    F: [m, 3] long
    with_gauss: 是否包含高斯曲率项
    """
    n = V.shape[0]

    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    e0 = v1 - v2
    e1 = v2 - v0
    e2 = v0 - v1

    l2_0 = (e0 * e0).sum(dim=1)
    l2_1 = (e1 * e1).sum(dim=1)
    l2_2 = (e2 * e2).sum(dim=1)

    dblA = torch.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)
    dblA = torch.clamp(dblA, min=_EPS_AREA)

    cot0 = (l2_1 + l2_2 - l2_0) / dblA * 0.25
    cot1 = (l2_2 + l2_0 - l2_1) / dblA * 0.25
    cot2 = (l2_0 + l2_1 - l2_2) / dblA * 0.25
    cot0 = torch.clamp(cot0, min=_COT_MIN, max=_COT_MAX)
    cot1 = torch.clamp(cot1, min=_COT_MIN, max=_COT_MAX)
    cot2 = torch.clamp(cot2, min=_COT_MIN, max=_COT_MAX)

    # 矩阵自由 Laplacian：非原地 index_add 链式累加，保证梯度完整
    LV = torch.zeros_like(V)
    w0 = cot0.unsqueeze(1)
    LV = LV.index_add_(0, F[:, 1], w0 * (v1 - v2))
    LV = LV.index_add_(0, F[:, 2], w0 * (v2 - v1))
    w1 = cot1.unsqueeze(1)
    LV = LV.index_add_(0, F[:, 2], w1 * (v2 - v0))
    LV = LV.index_add_(0, F[:, 0], w1 * (v0 - v2))
    w2 = cot2.unsqueeze(1)
    LV = LV.index_add_(0, F[:, 0], w2 * (v0 - v1))
    LV = LV.index_add_(0, F[:, 1], w2 * (v1 - v0))

    one_third_area = dblA / 6.0
    A = (
        torch.zeros(n, device=V.device, dtype=V.dtype).index_add_(0, F[:, 0], one_third_area)
        + torch.zeros(n, device=V.device, dtype=V.dtype).index_add_(0, F[:, 1], one_third_area)
        + torch.zeros(n, device=V.device, dtype=V.dtype).index_add_(0, F[:, 2], one_third_area)
    )
    A = torch.clamp(A, min=_EPS_AREA)
    inv_sqrt_A = torch.rsqrt(A + _EPS_DENOM)

    LV = LV * inv_sqrt_A.unsqueeze(1)
    E_mean = (LV * LV).sum()

    if not with_gauss:
        return E_mean

    a0, a1, a2 = _corner_angles_safe(v0, v1, v2)
    angle_sum = (
        torch.zeros(n, device=V.device, dtype=V.dtype).index_add_(0, F[:, 0], a0)
        + torch.zeros(n, device=V.device, dtype=V.dtype).index_add_(0, F[:, 1], a1)
        + torch.zeros(n, device=V.device, dtype=V.dtype).index_add_(0, F[:, 2], a2)
    )
    K = 2.0 * torch.pi - angle_sum
    E_gauss = -2.0 * K.sum()
    return E_mean + E_gauss


def thin_plate_energy_lowmem(
    V: torch.Tensor,
    F: torch.Tensor,
    with_gauss: bool = True,
):
    """与 thin_plate_energy 计算逻辑一致，保留接口兼容。"""
    return thin_plate_energy(V, F, with_gauss=with_gauss)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 简单网格：验证无数值异常且梯度有限
    V = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0], [0.5, 0.5, 1.0]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    F = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=torch.long, device=device)

    for name, fn in [("thin_plate_energy", thin_plate_energy), ("thin_plate_energy_lowmem", thin_plate_energy_lowmem)]:
        V2 = V.detach().clone().requires_grad_(True)
        E = fn(V2, F, with_gauss=True)
        assert torch.isfinite(E), f"{name}: E 含 nan/inf"
        E.backward()
        assert V2.grad is not None and torch.isfinite(V2.grad).all(), f"{name}: grad 含 nan/inf"
        print(f"{name}: E={E.item():.6f}, grad norm={V2.grad.norm().item():.6f}")

    # 近退化三角形 + 大坐标：应力测试
    V_degen = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1e-5, 0.0], [0.5, 0.5, 1.0]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    F_degen = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=torch.long, device=device)
    E_d = thin_plate_energy(V_degen, F_degen, with_gauss=True)
    E_d.backward()
    assert torch.isfinite(E_d) and torch.isfinite(V_degen.grad).all(), "近退化网格出现 nan/inf"
    print("近退化网格: E 与 grad 均有限")

    import os
    if os.path.isfile("bunny.ply"):
        import trimesh
        mesh = trimesh.load("bunny.ply", file_type="ply")
        V = torch.tensor(mesh.vertices, dtype=torch.float32, device=device, requires_grad=True)
        F = torch.tensor(mesh.faces, dtype=torch.long, device=device)
        E = thin_plate_energy(V, F)
        assert torch.isfinite(E), "bunny E 含 nan/inf"
        E.backward()
        assert torch.isfinite(V.grad).all(), "bunny grad 含 nan/inf"
        print("bunny: E=", E.item(), "grad finite:", torch.isfinite(V.grad).all().item())
    else:
        print("(无 bunny.ply，跳过网格测试)")
