import os
import argparse

import json
from collections import defaultdict, OrderedDict
import cv2
import numpy as np

from utils.rgb_to_json.data_utils import get_affine_transform, normalize, get_gt_class_keypoints_dict
from utils.rgb_to_json.infer_utils import get_final_preds, transform_preds, _coco_keypoint_results_all_category_kernel
from utils.rgb_to_json.nms import oks_nms

import tensorflow as tf

## Data Settings
IMAGE_SIZE = [288, 384] 
IMG_MEAN, IMG_STD =[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
C, S = np.array([959.5, 719.5]), np.array([11.368751, 15.158334]) # center, scale

## post-processing of prediction setiings 
target_weight = np.load('target_weight.npy')  # FIXED ???
gt_class_keypoints_dict = get_gt_class_keypoints_dict() # fixed dictionary
heatmap_height = 96 # = config.MODEL.HEATMAP_SIZE[1]
heatmap_width = 72 # \ config.MODEL.HEATMAP_SIZE[0]

num_samples = 1
NUM_JOINTS = 294
IN_VIS_THRE = 0.2
OKS_THRE = 0.9

def main(args) :
    img_file = args.image_path
    res_file = args.json_name
    print('img_file : ',img_file)
    print('res_file : ',res_file)
    
    
    ## Model(TFLite) Load
    tflite_file = args.model_path
    print('tflite_file : ',tflite_file)
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()
    
    ## Get input and output details(including the shape)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    
    ## Data load and transform
    data_numpy = cv2.imread(img_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    c, s, r = C, S, 0
    trans = get_affine_transform(c, s, r, IMAGE_SIZE)
    input_data = cv2.warpAffine(data_numpy, trans,
        (int(IMAGE_SIZE[0]), int(IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    input_data = normalize(input_data, IMG_MEAN, IMG_STD)
    input_data = np.array(input_data, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0).transpose(0,3,1,2)
    print('input_data.shape : ',input_data.shape)
    
    ## Model input and output
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    
    ## for final-result
    all_preds = np.zeros((num_samples, NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 7))
    
    ## output to json format
    idx = 0

    num_images = 1 # one image
    cat_ids = [1] # 반팔=1 , 
    c = np.expand_dims(C, 0)
    s = np.expand_dims(S, 0)
    score = [1] # fixed

    channel_mask = np.zeros_like(target_weight, dtype=np.float32)
    rg = gt_class_keypoints_dict[int(cat_ids[0])]
    index = np.array([list(range(rg[0], rg[1]))],dtype=np.int32).transpose(1,0)
    for i in index : # like pytorch _scatter -> ex) channel_mask[j].scatter_(0, index, 1)
        channel_mask[0][i] = [1]
    output = output * np.expand_dims(channel_mask, axis=3)


    preds_local, maxvals = get_final_preds(heatmap_height,heatmap_width, output, c, s)
    preds = preds_local.copy()
    for i in range(preds_local.shape[0]):
        preds[i] = transform_preds(
            preds_local[i], c[i], s[i], 
            [heatmap_width, heatmap_height])

    all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2] # from Pred
    all_preds[idx:idx + num_images, :, 2:3] = maxvals # from Pred

    # double check this all_boxes parts
    all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
    all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
    all_boxes[idx:idx + num_images, 4] = np.prod(s*200, 1)
    all_boxes[idx:idx + num_images, 5] = score
    all_boxes[idx:idx + num_images, 6] = np.array(cat_ids)

    _kpts = []
    for idx, kpt in enumerate(all_preds): # preds (2, 294, 3)
        _kpts.append({
            'keypoints': kpt, # (294, 3)
            'center': all_boxes[idx][0:2],
            'scale': all_boxes[idx][2:4],
            'area': all_boxes[idx][4],
            'score': all_boxes[idx][5],
            'category_id': all_boxes[idx][6],
    #         'image': int(img_path[idx][-10:-4]) # *** acceptable only for last 6-digit
            'image' : 1
        })

    kpts = defaultdict(list)
    for kpt in _kpts:
        kpts[kpt['image']].append(kpt)

    oks_nmsed_kpts = []
    for img in kpts.keys(): # each image 
        img_kpts = kpts[img]
        for n_p in img_kpts: # each items 
            box_score = n_p['score'] # score from 'meta'
            kpt_score = 0
            valid_num = 0
            for n_jt in range(0, NUM_JOINTS): # rach jt id 
                t_s = n_p['keypoints'][n_jt][2] # 3rd value 
                if t_s > IN_VIS_THRE: # if t_s > 0.2 
                    kpt_score = kpt_score + t_s
                    valid_num = valid_num + 1
            if valid_num != 0:
                kpt_score = kpt_score / valid_num
            n_p['score'] = kpt_score * box_score

        keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))], OKS_THRE)
        if len(keep) == 0:
            oks_nmsed_kpts.append(img_kpts)
        else:
            oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

    # json write
    results = _coco_keypoint_results_all_category_kernel(oks_nmsed_kpts, NUM_JOINTS)
    with open(res_file, 'w') as f:
        json.dump(results, f,sort_keys=True, indent=4)
    print(f'Successfully saved the model result for the RGB image as a json file ! ---> {res_file} ')


if __name__ == '__main__' :
    '''
    @@ 공지 @@
    당분간, 개발 실험에서
    Model은 hrnet, data는 validation_samples/our1.jpg 으로 고정합니다.
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '-ip', required=False, default='samples/point_cloud_sample1/rgb.jpg', help='image path')
    parser.add_argument('--model_path', '-mp', required=False, default='test_hrnet.tflite', help='model path')
    parser.add_argument('--json_name', '-jn', required=False, default='samples/point_cloud_sample1/estimated_kpt.json', help='model path')
    args = parser.parse_args()
    
    main(args)
    
    
    
    
    
    
    
    
    
    
    
