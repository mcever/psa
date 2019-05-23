"""
takes path to ground truths and predicitons and computes mIOU 
testing with png images where BW pixel values are class numbers
"""

from metrics import RunningConfusionMatrix as RCM

import scipy
import numpy as np
import argparse
import os

def compute_miou(gt_path, pred_path):
    CM = RCM(list(range(21)))

    preds = os.listdir(pred_path)
    print('n preds: {}'.format(len(preds)))
    gts = os.listdir(gt_path)
    for fname in preds:
        # open pred and gt as numpy arrays, flatten them
        p_img = scipy.misc.imread(os.path.join(pred_path, fname))
        """
        if np.all(p_img == np.zeros(p_img.shape) ):
            print('frt')
            continue
        """
        gt_img = scipy.misc.imread(os.path.join(gt_path, fname))
        CM.update_matrix(gt_img.flatten(), p_img.flatten())

    miou = CM.compute_current_mean_intersection_over_union()
    return miou


if __name__ == "__main__":

    VOC_HOME = '/media/ssd1/austin/datasets/VOC/VOCdevkit/VOC2012/AugSegClass255'

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", default=VOC_HOME, type=str)
    parser.add_argument("--pred_path", default='out_cam_pred', type=str)
    args = parser.parse_args()

    miou = compute_miou(args.gt_path, args.pred_path)
    print('miou:{}'.format(miou))

