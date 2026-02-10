import torch 
import numpy as np


def output_multiple_XYZN(
    face2targetN: dict, 
    face_centroid: torch.Tensor, 
    filename: str = "all_target_points.xyz"
):
    print("open")

    # 将所有数据拼接成 numpy 数组
    all_points = []
    all_normals = []
    for key, values in face2targetN.items():
        if len(values) == 0:
            continue
        p = face_centroid[key].cpu().numpy()  # [3]
        n = np.stack(values, axis=0)          # [M,3]
        points = np.tile(p, (n.shape[0], 1))  # 每个面片的质心重复 M 次
        all_points.append(points)
        all_normals.append(n)

    # 拼接所有
    all_points = np.concatenate(all_points, axis=0)  # [total,3]
    all_normals = np.concatenate(all_normals, axis=0)  # [total,3]

    # 合并 XYZ + NXNYNZ
    data = np.hstack([all_points, all_normals])  # [total,6]

    # 一次性写入文件
    np.savetxt(filename, data, fmt="%.6f")
    print(f"Saved points to {filename}")
