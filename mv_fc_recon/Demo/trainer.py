import sys
sys.path.append('../../zhaozihan/camera-control')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import pickle

from mv_fc_recon.Method.time import getCurrentTime
from mv_fc_recon.Module.trainer import Trainer


def demo():
    # 小妖怪头
    # shape_id = "003fe719725150960b14cdd6644afe5533743ef63d3767c495ab76a6384a9b2f"
    # 女人上半身
    # shape_id = "017c6e8b81a17c298baf2aba24fd62fa5a992ba8346bc86b2b5008caf1478873"
    # 长发男人头
    shape_id = "0228c5cdba8393cd4d947ac2e915f769f684c73b87e6939c129611ba665cafcb"

    # shape_id = "bunny" 

    device = 'cuda:0'

    home = os.environ['HOME']

    data_folder = home + "/chLi/Dataset/pixel_align/" + shape_id + "/"

    camera_pkl_file_path = data_folder + 'camera.pkl'
    gen_mesh_file_path = data_folder + "stage2_192_n_d2_d4_d8_d16.ply"
    # gen_mesh_file_path = data_folder + "bunny.ply"
    # print(gen_mesh_file_path)
    # gen_mesh_file_path = data_folder + "gt_normalized.ply"
    log_dir = data_folder + 'mv-fc-recon/logs/' + getCurrentTime() + '/'

    assert os.path.exists(camera_pkl_file_path), f"camera.pkl not found at {camera_pkl_file_path}"
    with open(camera_pkl_file_path, 'rb') as f:
        camera_list = pickle.load(f)

    # print("zzh: ", camera_list) 
    # return  
    fitting_mesh = Trainer.fitImagesWithSDFLoss(
        camera_list=camera_list,  ##各个位置的相机参数
        mesh=gen_mesh_file_path,  ##输入模型
        resolution=192,  ##192
        device=device,
        num_iterations=200, ##200次迭代 
        log_interval=5,
        log_dir=log_dir,
    )

    # print("zzh: ", fitting_mesh) 
    fitting_mesh.export(log_dir + 'fc_fitting_mesh.ply')
    return True
