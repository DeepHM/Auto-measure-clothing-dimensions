import os
import os.path as osp
import argparse
import copy

import math
import numpy as np
from PIL import Image
import cv2
from plyfile import PlyData
import json

from utils.json_to_landmark.pyutils import _kps1d_to_2d, getImageRatio, _kps_downscale, get_proj_depth, get_distance
from utils.json_to_landmark.size_info import getSizingPts

''' hyper-params '''
deg = 0.275

def main(args) :
    root = args.root
    print('root :',root)
    
    ''' File Load '''
    predicted_json_file = osp.join(root, 'estimated_kpt.json')
    print('predicted_json_file : ',predicted_json_file)
    with open(predicted_json_file, 'r') as f:
        predicted_json = json.load(f)
        
    img_path = f'{root}/rgb.jpg'
    ply_path = f'{root}/ply.ply'
    print('img_path :',img_path)
    print('ply_path :',ply_path)

    image = Image.open(img_path) 
    ply = PlyData.read(ply_path) 
    print('image size :',image.size)
    ''' '''
    
    ''' json to Landmark position & length(cm) calculation '''
    measure_index, measure_points = getSizingPts(1) # 1: 반팔
    w_r, h_r = getImageRatio(image, ply)

    predicted = predicted_json[0]
    pred_kpt1d = predicted['keypoints']
    kps_arr = _kps1d_to_2d(pred_kpt1d)
    kps_arr = _kps_downscale(kps_arr, (w_r, h_r)) 
    kps_dict = {i+1: arr for i, arr in enumerate(kps_arr)}
    ply_dpt = get_proj_depth(ply)  # projected depth array

    result = {name:{'pt1':None, 'pt2':None, 'cm':None} for name in measure_index.values()}
    for a, name in measure_index.items() : 
        pt1, pt2 = [tuple(kps_dict[pt_key].astype(int)) for pt_key in measure_points[a]]
        result[name]['depth_pt1'] = int(pt1[0]) , int(pt1[1])
        result[name]['depth_pt2'] = int(pt2[0]) , int(pt2[1])
        result[name]['pt1'] = int(pt1[0]*w_r), int(pt1[1]*h_r)
        result[name]['pt2'] = int(pt2[0]*w_r), int(pt2[1]*h_r)
        
        size_ = get_distance(pt1, pt2, ply_dpt, deg=deg)
        result[name]['cm'] = round(size_, 2)
        
    ''' '''
    
    ''' Draw a circle, line and length on an image '''
    img_arr = np.asarray(image)
    for r in result : 
        pt1, pt2 = result[r]['pt1'] , result[r]['pt2']
        cm = result[r]['cm']

        for c in [pt1, pt2] :
            cv2.circle(img_arr,
                    c,
                    5,
                    (255,0,0),
                    thickness=-1
                  )
        cv2.line(img_arr,
            pt1,
            pt2,
            (255,0,0), 
            thickness=2, 
            lineType=cv2.LINE_AA)
        cv2.putText(img_arr, r, (pt2[0]-100,pt2[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        if r == 'Waist-length' :
             cv2.putText(img_arr, f'({cm}cm)', (pt2[0]+110,pt2[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        elif r == 'Total-length' :
            cv2.putText(img_arr, f'({cm}cm)', (pt2[0]+100,pt2[1]+40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        else :
            cv2.putText(img_arr, f'({cm}cm)', (pt2[0]-100,pt2[1]+80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    ''' '''
    
    ''' result Image and json save'''
    print(result)
    with open(f'{root}/result_kpt_v1.json', 'w') as f:
        json.dump(result, f)
    
    result_img = Image.fromarray(img_arr)
    result_img.save(f'{root}/result_image_v1.png', 'png')
    
    print('Finish !')
    print(f'Result json save --> {root}/result_kpt_v1.png',)
    print(f'Result Image save --> {root}/result_image.png',)


if __name__ == '__main__' :    
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', '-r', required=False, default='samples/point_cloud_sample1', help='file root')
    args = parser.parse_args()
    
    main(args)
    
    
    
    
    
    
    
    
    
    
    
