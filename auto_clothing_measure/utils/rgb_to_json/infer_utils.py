# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
from .data_utils import get_affine_transform


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(heatmap_height,heatmap_width, output, center, scale, coord_heatmaps=None):
    # heatmap_height = config.MODEL.HEATMAP_SIZE[1]
    # heatmap_width = config.MODEL.HEATMAP_SIZE[0]
    
    batch_heatmaps = output
    coords, maxvals = get_max_preds(batch_heatmaps)
    
    ## post-processing ## 
    # if config.TEST.POST_PROCESS:  <- True
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(math.floor(coords[n][p][0] + 0.5))
            py = int(math.floor(coords[n][p][1] + 0.5))
            if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                diff = np.array(
                    [
                        hm[py][px+1] - hm[py][px-1],
                        hm[py+1][px]-hm[py-1][px]
                    ]
                )
                coords[n][p] += np.sign(diff) * .25

    return coords, maxvals

def _coco_keypoint_results_all_category_kernel(keypoints, num_joints): # *** 
    # cat_id = data_pack['cat_id']

    keypoints = keypoints
    cat_results = []

    for img_kpts in keypoints: # each image 
        if len(img_kpts) == 0:
            continue

        for k in range(len(img_kpts)): # each item 
            image_id = img_kpts[k]['image'] # image_id 
            cat_id = img_kpts[k]['category_id']
            score = img_kpts[k]['score'] # mean estimation score

            _key_points = np.array([img_kpts[k]['keypoints']
                                    for k in range(len(img_kpts))]) # (2, 294, 3) 
            key_points = np.zeros((_key_points.shape[0], num_joints * 3), dtype=np.float32) # (2, 882) 

            for ipt in range(num_joints):
                key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
                key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
                key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

            result = [
                {
                    'image_id': image_id,
                    'category_id': cat_id,
                    # 'keypoints': list(key_points[k]),
                    'keypoints': key_points[k].tolist(),
                    'score': score,
                    # 'score': img_kpts[k]['score'],
                    # 'center': list(img_kpts[k]['center']),
                    # 'scale': list(img_kpts[k]['scale'])
                }
            ]
            cat_results.extend(result)

    return cat_results




