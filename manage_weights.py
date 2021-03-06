# continuously check for new weights, run inference
# check miou, and save

import os
import time
import pickle
import subprocess
from compute_miou import compute_miou

run_num = '0514'
weights_dir = 'caff_psa'

while 1:
    files = os.listdir(weights_dir)
    ep_2_fname = {}
    for fi in files:
        if '.pth' in fi:
            if fi.split('_')[2] == run_num:
                try:
                    ep = fi.split('_')[4].split('.')[0]
                    ep_2_fname[ep] = fi
                except:
                    pass

    try:
        with open('{}_miou.p'.format(run_num), 'rb') as f:
            ep_2_miou = pickle.load(f)
    except FileNotFoundError:
        ep_2_miou = {}

    eps = list(ep_2_fname.keys())
    eps.sort(key=float, reverse=True)
    myep = -1

    for ep in eps:
        if ep not in ep_2_miou:
            ep_2_miou[ep] = -1
            with open('{}_miou.p'.format(run_num), 'wb') as f:
                pickle.dump(ep_2_miou, f)
            myep = ep
            break

    if myep == -1:
        continue # continue the while 1, let's check all the weights again

    # otherwise, run inference

    start= time.time()
    print(start)
    cmd = 'CUDA_VISBILE_DEVICES=1,2 python3 infer_cls.py --infer_list voc12/val.txt --voc12_root /media/ssd1/austin/datasets/VOC/VOCdevkit/VOC2012 --network network.resnet38_cls --weights {} --out_cam_pred out_cam_pred'.format( os.path.join(weights_dir, ep_2_fname[myep]) )
    print(cmd)
    val_out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    end = time.time()
    print(end)
    print('elapsed: {}'.format(end - start))

    # then compute miou
    gt_path = '/media/ssd1/austin/datasets/VOC/VOCdevkit/VOC2012/AugSegClass'
    pred_path = 'out_cam_pred'
    miou = compute_miou(gt_path, pred_path)
    print(miou, myep)

    # save miou to pickle
    ep_2_miou[myep] = miou
    print(ep_2_miou)
    with open('{}_miou.p'.format(run_num), 'wb') as f:
        ep_2_miou = pickle.dump(ep_2_miou, f)

