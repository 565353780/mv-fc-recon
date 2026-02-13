import sys
sys.path.append('../camera-control')
sys.path.append('../flexi-cubes')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import pickle

from camera_control.Method.io import loadMeshFile
from camera_control.Method.mesh import normalizeMesh
from camera_control.Module.mesh_renderer import MeshRenderer

from mv_fc_recon.Method.time import getCurrentTime
from mv_fc_recon.Module.trainer import Trainer


def demo():
    # 小妖怪头
    # shape_id = "003fe719725150960b14cdd6644afe5533743ef63d3767c495ab76a6384a9b2f"
    # 女人上半身
    # shape_id = "017c6e8b81a17c298baf2aba24fd62fa5a992ba8346bc86b2b5008caf1478873"
    # 长发男人头
    shape_id = "0228c5cdba8393cd4d947ac2e915f769f684c73b87e6939c129611ba665cafcb"

    device = 'cuda:0'

    home = os.environ['HOME']

    data_folder_path = home + "/chLi/Dataset/pixel_align/" + shape_id + "/"

    camera_pkl_file_path = data_folder_path + 'camera_fc.pkl'
    if not os.path.exists(camera_pkl_file_path):
        normalized_gt_mesh_file_path = data_folder_path + 'gt_normalized.ply'
        if not os.path.exists(normalized_gt_mesh_file_path):
            gt_mesh_file_path = data_folder_path + 'gt.glb'
            mesh = loadMeshFile(gt_mesh_file_path)
            mesh = normalizeMesh(mesh, target_length=0.99)
            mesh.export(normalized_gt_mesh_file_path)

        mesh = loadMeshFile(normalized_gt_mesh_file_path)

        camera_list = MeshRenderer.sampleRenderData(
            mesh,
            camera_num=50,
            camera_dist=1.6,
            width=1024,
            height=1024,
            fx=1407,
            fy=1407,
            bg_color=[255, 255, 255],
            device=device,
        )
        with open(camera_pkl_file_path, "wb") as f:
            pickle.dump(camera_list, f)

    gen_mesh_file_path = data_folder_path + "stage2_full.glb"
    log_dir = data_folder_path + 'mv-fc-recon/logs/' + getCurrentTime() + '/'

    assert os.path.exists(camera_pkl_file_path), f"camera.pkl not found at {camera_pkl_file_path}"
    with open(camera_pkl_file_path, 'rb') as f:
        camera_list = pickle.load(f)

    mesh = loadMeshFile(gen_mesh_file_path)
    mesh = normalizeMesh(mesh, target_length=0.99)
    fitting_mesh = Trainer.fitImagesWithSDFLoss(
        camera_list=camera_list,
        mesh=mesh,
        resolution=192,
        device=device,
        num_iterations=200,
        log_interval=5,
        log_dir=log_dir,
    )

    # fitting_mesh.export(log_dir + 'fc_fitting_mesh.ply')
    fitting_mesh.export( shape_id + '_fc_fitting_mesh.ply')
    print("[result] ", shape_id + '_fc_fitting_mesh.ply') 
    return True
