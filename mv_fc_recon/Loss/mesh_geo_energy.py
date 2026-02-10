from typing import Optional, Tuple 
import torch 
import trimesh 
from igl import cotmatrix , read_triangle_mesh 


import torch


def thin_plate_energy_lowmem(V: torch.Tensor,
                             F: torch.Tensor,
                             with_gauss: bool = True):
    """
    Memory-efficient thin plate energy.

    V: [n,3] requires_grad=True
    F: [m,3] long
    """

    n = V.shape[0]

    # -------------------------------------------------
    # 1. Gather triangle vertices (only once)
    # -------------------------------------------------
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]

    # -------------------------------------------------
    # 2. Geometry per triangle
    # -------------------------------------------------
    e0 = v1 - v2
    e1 = v2 - v0
    e2 = v0 - v1

    l2_0 = (e0 * e0).sum(dim=1)
    l2_1 = (e1 * e1).sum(dim=1)
    l2_2 = (e2 * e2).sum(dim=1)

    dblA = torch.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)
    dblA = torch.clamp(dblA, min=1e-6)
    # dblA = dblA + 1e-12

    cot0 = (l2_1 + l2_2 - l2_0) / dblA * 0.25
    cot1 = (l2_2 + l2_0 - l2_1) / dblA * 0.25
    cot2 = (l2_0 + l2_1 - l2_2) / dblA * 0.25

    cot0 = torch.clamp(cot0, max=1e3)  ##computation safe 
    cot1 = torch.clamp(cot1, max=1e3)  ##computation safe 
    cot2 = torch.clamp(cot2, max=1e3)  ##computation safe 

    # -------------------------------------------------
    # 3. Matrix-free Laplacian: LV = L V
    # -------------------------------------------------
    LV = torch.zeros_like(V)

    # edge (1,2)
    w = cot0.unsqueeze(1)
    LV.index_add_(0, F[:,1], w * (v1 - v2))
    LV.index_add_(0, F[:,2], w * (v2 - v1))

    # edge (2,0)
    w = cot1.unsqueeze(1)
    LV.index_add_(0, F[:,2], w * (v2 - v0))
    LV.index_add_(0, F[:,0], w * (v0 - v2))

    # edge (0,1)
    w = cot2.unsqueeze(1)
    LV.index_add_(0, F[:,0], w * (v0 - v1))
    LV.index_add_(0, F[:,1], w * (v1 - v0))

    # -------------------------------------------------
    # 4. Barycentric vertex area
    # -------------------------------------------------
    A = torch.zeros(n, device=V.device, dtype=V.dtype)
    A.index_add_(0, F[:,0], dblA / 6.0)
    A.index_add_(0, F[:,1], dblA / 6.0)
    A.index_add_(0, F[:,2], dblA / 6.0)

    A = torch.clamp(A, min=1e-6)
    inv_sqrt_A = torch.rsqrt(A)  ##computation safe 
    # inv_sqrt_A = torch.rsqrt(A + 1e-12)

    LV *= inv_sqrt_A.unsqueeze(1)

    E_mean = (LV * LV).sum()

    # -------------------------------------------------
    # 5. Gaussian term (optional)
    # -------------------------------------------------
    if with_gauss:

        # angle via normalized dot (no nested function)
        # def angle(u, v):
        #     dot = (u * v).sum(dim=1)
        #     nu = torch.norm(u, dim=1)
        #     nv = torch.norm(v, dim=1)
        #     cos = dot / (nu * nv + 1e-12)
        #     cos = torch.clamp(cos, -1.0, 1.0)
        #     return torch.acos(cos)
        def angle(u, v):
            cross = torch.norm(torch.cross(u, v, dim=1), dim=1)
            dot = (u * v).sum(dim=1)
            return torch.atan2(cross, dot)  ##computation safe 

        a0 = angle(v1 - v0, v2 - v0)
        a1 = angle(v2 - v1, v0 - v1)
        a2 = angle(v0 - v2, v1 - v2)

        angle_sum = torch.zeros(n, device=V.device, dtype=V.dtype)
        angle_sum.index_add_(0, F[:,0], a0)
        angle_sum.index_add_(0, F[:,1], a1)
        angle_sum.index_add_(0, F[:,2], a2)

        K = 2.0 * torch.pi - angle_sum
        E_gauss = -2.0 * K.sum()
        
        # print("e mean: ", E_mean, "  |  e gauss: ", E_gauss ) 
        return E_mean + E_gauss

    return E_mean





def thin_plate_energy(V: torch.Tensor, F: torch.Tensor):
    """
    Compute the discrete thin-plate energy for a mesh:
        E = ||MLV||^2 - 2\sum_i [ (2pi - \sum_j theta_j) ]
    
    Args:
        V: [n,3] float tensor of vertices, requires_grad=True
        F: [m,3] long tensor of faces
    Returns:
        E: scalar tensor
    """
    n = V.shape[0]
    m = F.shape[0]
    
    # ---------------------------
    # 1. compute cotangent Laplacian
    # ---------------------------
    v0 = V[F[:,0], :]
    v1 = V[F[:,1], :]
    v2 = V[F[:,2], :]
    
    # edge vectors opposite each vertex
    e0 = v1 - v2  # opposite v0
    e1 = v2 - v0  # opposite v1
    e2 = v0 - v1  # opposite v2
    
    # squared lengths
    l2_0 = (e0**2).sum(dim=1)
    l2_1 = (e1**2).sum(dim=1)
    l2_2 = (e2**2).sum(dim=1)
    
    # double area
    dblA = torch.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)
    
    # cotangents per corner
    cot0 = (l2_1 + l2_2 - l2_0) / dblA / 4.0
    cot1 = (l2_2 + l2_0 - l2_1) / dblA / 4.0
    cot2 = (l2_0 + l2_1 - l2_2) / dblA / 4.0
    cot = torch.stack([cot0, cot1, cot2], dim=1)  # [m,3]


    src = torch.stack([
        F[:,1], F[:,2],
        F[:,2], F[:,0],
        F[:,0], F[:,1]
    ], dim=1)

    dst = torch.stack([
        F[:,2], F[:,1],
        F[:,0], F[:,2],
        F[:,1], F[:,0]
    ], dim=1)

    w = torch.stack([
        cot[:,0], cot[:,0],
        cot[:,1], cot[:,1],
        cot[:,2], cot[:,2]
    ], dim=1)

    # reshape to 1D
    I_off = src.reshape(-1)
    J_off = dst.reshape(-1)
    W_off = w.reshape(-1)

    # diagonal accumulation
    diag = torch.zeros(n, device=V.device, dtype=V.dtype) 
    diag.index_add_(0, I_off, -W_off) 

    # assemble
    I = torch.cat([I_off, torch.arange(n, device=V.device)])
    J = torch.cat([J_off, torch.arange(n, device=V.device)])
    W = torch.cat([W_off, diag])

    L = torch.sparse_coo_tensor(
        torch.stack([I, J]),
        W,
        size=(n, n)
    ).coalesce()


    # ---------------------------
    # 2. per-vertex barycentric area for W (voronoi area比较复杂不易计算，用简易的 barycentric area)
    # ---------------------------
    A = torch.zeros(n, device=V.device, dtype=V.dtype)
    A.index_add_(0, F[:,0], dblA/6)
    A.index_add_(0, F[:,1], dblA/6)
    A.index_add_(0, F[:,2], dblA/6)
    
    W_diag = 1.0 / torch.sqrt(A + 1e-12)  # avoid div0
    
    # ---------------------------
    # 3. mean curvature term
    # ---------------------------
    LV = torch.sparse.mm(L, V)       # [n,3]
    LV = LV * W_diag[:, None] 
    E_mean = (LV ** 2).sum()
    
    # ---------------------------
    # 4. gaussian curvature term
    # ---------------------------
    # compute angles per corner
    def corner_angles(v0,v1,v2):
        a0 = torch.acos(
            torch.clamp(((v1-v0)*(v2-v0)).sum(dim=1) / 
                        (torch.norm(v1-v0,dim=1)*torch.norm(v2-v0,dim=1)+1e-12), -1.0,1.0)
        )
        a1 = torch.acos(
            torch.clamp(((v2-v1)*(v0-v1)).sum(dim=1) / 
                        (torch.norm(v2-v1,dim=1)*torch.norm(v0-v1,dim=1)+1e-12), -1.0,1.0)
        )
        a2 = torch.acos(
            torch.clamp(((v0-v2)*(v1-v2)).sum(dim=1) / 
                        (torch.norm(v0-v2,dim=1)*torch.norm(v1-v2,dim=1)+1e-12), -1.0,1.0)
        )
        return a0,a1,a2
    
    a0,a1,a2 = corner_angles(v0,v1,v2)

    angle_sum = torch.zeros(n, device=V.device, dtype=V.dtype)
    angle_sum.index_add_(0, F[:,0], a0)
    angle_sum.index_add_(0, F[:,1], a1)
    angle_sum.index_add_(0, F[:,2], a2)

    K = 2*torch.pi - angle_sum    # 每个顶点只加一次 2pi
    E_gauss = -2.0 * K.sum() 
    
    # ---------------------------
    # 5. total energy
    # ---------------------------
    E = E_mean + E_gauss
    # print("e mean: ", E_mean, "  |  e gauss: ", E_gauss ) 
    
    return E







if __name__ == "__main__": 
    # V, F = read_triangle_mesh("./bunny.ply")
    # L = cotmatrix(V, F)
    
    mesh = trimesh.load("bunny.ply", file_type="ply") 
    print(mesh.vertices.shape, mesh.faces.shape )
    device = 'cuda'  # 或 'cpu'

    V = torch.tensor(mesh.vertices, dtype=torch.float32, device=device, requires_grad=True)
    F = torch.tensor(mesh.faces, dtype=torch.long, device=device, requires_grad=False )
    

    energy = thin_plate_energy_lowmem(V, F) 

    print("energy: ", energy )

    print(V.grad) ## None 
    energy.backward() 
    print(V.grad) ## 
 

