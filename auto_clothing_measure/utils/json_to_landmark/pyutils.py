import copy
import numpy as np

def getImageRatio(image, ply): 
    # Image size 
    w, h = image.size

    # ply cx, cy 
    # vert_ = ply['vertex']
    cx_w = ply['vertex']['cx'].max() + 1    # 256.0 
    cy_h = ply['vertex']['cy'].max() + 1    # 192.0 

    # resizing ratio 
    w_r = w/cx_w
    h_r = h/cy_h

    return(w_r, h_r)

def getImageRatio_V2(image, shape): 
    w, h = image.size    # (1920, 1440) 
    
    cx_w = shape[1] # 256.0 
    cy_h = shape[0] # 192.0 

    w_r = w/cx_w
    h_r = h/cy_h

    return (w_r, h_r)


def _kps1d_to_2d(kps1d):
    kps = copy.deepcopy(kps1d)

    kps_num = divmod(len(kps), 3)[0]

    kps_2d_ls = [[kps[kp_i*3+0], kps[kp_i*3+1]] for kp_i in range(kps_num) if kps[kp_i*3+2]] # List
    kps_2d_arr = np.array(kps_2d_ls)

    return kps_2d_arr

def _kps_downscale(kps_arr_row, resize_r): # ver.02 ***    
    kps_arr = copy.deepcopy(kps_arr_row)
    w_r, h_r = resize_r 
    
    kps_arr[:, 0] = kps_arr[:, 0]/w_r
    kps_arr[:, 1] = kps_arr[:, 1]/h_r

    return(kps_arr)

def get_proj_depth(ply): 
    vert_ = ply['vertex']
    x_loc = vert_['cx'].max() + 1    # 256.0 
    y_loc = vert_['cy'].max() + 1    # 192.0 

    # proj_depth = np.full((int(x_loc), int(y_loc)), -1, dtype=np.float32) 
    proj_depth = np.full((int(x_loc), int(y_loc)), -1, dtype=np.float32) 

    for i in range(vert_.count): 
        proj_depth[int(vert_['cx'][i]), int(vert_['cy'][i])] = vert_['depth'][i] 

    return(proj_depth) 

def get_proj_depth_V2(ply, shape) :
    vertices = np.zeros(shape=[ply['vertex'].count, 8], dtype=np.float32)
    vertices[:, 0] = ply['vertex'].data['x']
    vertices[:, 1] = ply['vertex'].data['y']
    vertices[:, 2] = ply['vertex'].data['z']
    vertices[:, 3] = ply['vertex'].data['r']
    vertices[:, 4] = ply['vertex'].data['g']
    vertices[:, 5] = ply['vertex'].data['b']
    vertices[:, 6] = ply['vertex'].data['x_coord']
    vertices[:, 7] = ply['vertex'].data['y_coord']
    
    proj = np.full((3, shape[0], shape[1]), -1,dtype=np.float32)
    for k in range(len(vertices)):
        proj[0, int(vertices[:, 7][k]), int(vertices[:, 6][k])] = vertices[k, 0]
        proj[1, int(vertices[:, 7][k]), int(vertices[:, 6][k])] = vertices[k, 1]
        proj[2, int(vertices[:, 7][k]), int(vertices[:, 6][k])] = vertices[k, 2]
    proj = proj.transpose(1,2,0)
    
    return proj



def get_distance(pt1, pt2, ply_dpt, deg=0.28): 
    # # 1. 점1. 점2 좌표 받기 
    # # ----------------------- 
    # pt1 = (55, 151) # (w, h) up-left
    # pt2 = (55, 34) # (w, h) up-right

    # 2. depth 데이터 받고 , + 각 선1, 선2에 depth 받고 
    dpt_arr = ply_dpt.copy()
    # 방향! 신경쓰기!! 
    # >>> dpt_arr.shape # (256, 192)
    # ----------------------- 
    d1 = dpt_arr[pt1]
    d2 = dpt_arr[pt2]

    # 3. 점간 기본 사잇각 arg로 받기 
    # deg = 0.26 # degree 0.258
    # import numpy as np
    # ----------------------- 
    rad = np.deg2rad(deg) # ex.) 0.004363323129985824

    # 4. 점1, 점2 사이 각 계산 
    # ----------------------- 
    px_d = np.sqrt(np.square([pt1[0]-pt2[0], pt1[1]-pt2[1]]).sum()) # 점 사이 (픽셀)거리 
    px_rad = px_d*rad # 점사이 추정 각 (라디안 값)

    # 5. 사이각에 대한 코사인 값 계산 -> 사이거리 계산(depth)
    # ----------------------- 
    ss = np.square([d1*np.sin(px_rad), d2 - d1*np.cos(px_rad)]).sum()
    size_d = np.sqrt(ss)

    # (6.) cm 측도로 반환 
    # ----------------------- 
    # print(f" * size: {size_d*100: .4f} cm")
    return(size_d*100) # *100: (mm) -> (cm) 


