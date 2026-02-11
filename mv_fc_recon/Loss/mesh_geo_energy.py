import os
import torch
import trimesh

_EPS = 1e-12
_EPS_AREA = 1e-10
_COT_MIN = -1e6
_COT_MAX = 1e6


# ---------------------------------------------------
# Stable corner angles (atan2, igl-consistent)
# ---------------------------------------------------
def _corner_angles(v0, v1, v2):
    def angle(a, b):
        cross = torch.linalg.norm(torch.cross(a, b, dim=1), dim=1)
        dot = (a * b).sum(dim=1)
        return torch.atan2(cross, dot + _EPS)

    return (
        angle(v1 - v0, v2 - v0),
        angle(v2 - v1, v0 - v1),
        angle(v0 - v2, v1 - v2),
    )


# ---------------------------------------------------
# Cot Laplacian (matrix-free, igl-style)
# ---------------------------------------------------
def _cot_laplacian(V, F):
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    e0 = v1 - v2
    e1 = v2 - v0
    e2 = v0 - v1

    l2_0 = (e0 * e0).sum(dim=1)
    l2_1 = (e1 * e1).sum(dim=1)
    l2_2 = (e2 * e2).sum(dim=1)

    dblA = torch.linalg.norm(
        torch.cross(v1 - v0, v2 - v0, dim=1), dim=1
    )
    dblA = torch.clamp(dblA, min=_EPS_AREA)

    cot0 = (l2_1 + l2_2 - l2_0) / (4.0 * dblA)
    cot1 = (l2_2 + l2_0 - l2_1) / (4.0 * dblA)
    cot2 = (l2_0 + l2_1 - l2_2) / (4.0 * dblA)

    cot0 = torch.clamp(cot0, _COT_MIN, _COT_MAX)
    cot1 = torch.clamp(cot1, _COT_MIN, _COT_MAX)
    cot2 = torch.clamp(cot2, _COT_MIN, _COT_MAX)

    Lx = torch.zeros_like(V)

    Lx = Lx.index_add_(0, F[:, 1], cot0[:, None] * (v1 - v2))
    Lx = Lx.index_add_(0, F[:, 2], cot0[:, None] * (v2 - v1))

    Lx = Lx.index_add_(0, F[:, 2], cot1[:, None] * (v2 - v0))
    Lx = Lx.index_add_(0, F[:, 0], cot1[:, None] * (v0 - v2))

    Lx = Lx.index_add_(0, F[:, 0], cot2[:, None] * (v0 - v1))
    Lx = Lx.index_add_(0, F[:, 1], cot2[:, None] * (v1 - v0))

    return Lx, dblA


# ---------------------------------------------------
# Lumped vertex areas (igl-style barycentric)
# ---------------------------------------------------
def _vertex_areas(dblA, F, n):
    A = torch.zeros(n, device=dblA.device, dtype=dblA.dtype)
    tri_area = dblA / 6.0
    for i in range(3):
        A = A.index_add_(0, F[:, i], tri_area)
    return torch.clamp(A, min=_EPS_AREA)


# ---------------------------------------------------
# FINAL Thin Plate Energy (igl-aligned)
# ---------------------------------------------------
def thin_plate_energy(
    V: torch.Tensor,
    F: torch.Tensor,
    w_mean: float = 1.0,
    w_gauss: float = 1.0,
):
    """
    Thin-plate bending energy strictly aligned with libigl definitions.

    E = Σ_i A_i ( w_mean * ||H_i||^2 - 2 w_gauss * K_i )

    V : [n,3], requires_grad=True
    F : [m,3], long
    """

    n = V.shape[0]

    # ---- Laplacian
    Lx, dblA = _cot_laplacian(V, F)
    A = _vertex_areas(dblA, F, n)

    # ---- Mean curvature normal (igl definition)
    Hn = -Lx / A[:, None]

    # ---- Mean curvature energy
    E_mean = (A[:, None] * (Hn * Hn)).sum()

    # ---- Gaussian curvature
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    a0, a1, a2 = _corner_angles(v0, v1, v2)

    angle_sum = torch.zeros(n, device=V.device, dtype=V.dtype)
    angle_sum = angle_sum.index_add_(0, F[:, 0], a0)
    angle_sum = angle_sum.index_add_(0, F[:, 1], a1)
    angle_sum = angle_sum.index_add_(0, F[:, 2], a2)

    K = (2.0 * torch.pi - angle_sum) / A

    # ---- Gaussian energy
    E_gauss = (A * K).sum()

    # ---- Thin plate energy
    energy = w_mean * E_mean - 2.0 * w_gauss * E_gauss

    return energy

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

    V2 = V.detach().clone().requires_grad_(True)
    E = thin_plate_energy(V2, F)
    assert torch.isfinite(E), "E 含 nan/inf"
    E.backward()
    assert V2.grad is not None and torch.isfinite(V2.grad).all(), f"grad 含 nan/inf"
    print(f"E={E.item():.6f}, grad norm={V2.grad.norm().item():.6f}")

    # 近退化三角形 + 大坐标：应力测试
    V_degen = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1e-5, 0.0], [0.5, 0.5, 1.0]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    F_degen = torch.tensor([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=torch.long, device=device)
    E_d = thin_plate_energy(V_degen, F_degen)
    E_d.backward()
    assert torch.isfinite(E_d) and torch.isfinite(V_degen.grad).all(), "近退化网格出现 nan/inf"
    print("近退化网格: E 与 grad 均有限")

    mesh_file_path = os.environ['HOME'] + "/chLi/Dataset/bunny.ply"
    if os.path.isfile(mesh_file_path):
        mesh = trimesh.load(mesh_file_path)
        V = torch.tensor(mesh.vertices, dtype=torch.float32, device=device, requires_grad=True)
        F = torch.tensor(mesh.faces, dtype=torch.long, device=device)
        E = thin_plate_energy(V, F)
        assert torch.isfinite(E), "bunny E 含 nan/inf"
        E.backward()
        assert torch.isfinite(V.grad).all(), "bunny grad 含 nan/inf"
        print("bunny: E=", E.item(), "grad finite:", torch.isfinite(V.grad).all().item())
    else:
        print("(无 bunny.ply，跳过网格测试)")
