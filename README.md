# Auto-measure-clothing-dimensions
Deep learning based automatic clothing length measurement project using PointCloud and RGB images

# Purpose of the project
Using RGB images and Point Cloud data, build a deep learning-based automatic clothing length measurement system that can operate on-device inside a mobile device.

# Overview
<img src="https://user-images.githubusercontent.com/37736774/215036841-c9c5aad5-bcf0-4693-a067-b5d56d18f0cb.png" width="800" height="400"/>

   
   
<br/>
<br/>


# code process

### Preparation 

- Creating a virtual environment and registering the virtual environment in Jupyter

```
conda create -n measure python=3.7
conda activate measure
pip install jupyter
python -m ipykernel install --user --name measure --display-name "measure"
```

- install onnx_tensorflow & pytorch

```
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
cd install_onnx_tensorflow
pip install -e .
```

- Install the rest of the packages

```
cd ..
bash install_packages.sh
```

* Upload ***pose_hrnet-w48_384x288-deepfashion2_mAP_0.7017.pth*** to fashion_landmark.

   * download : (https://shanghaitecheducn-my.sharepoint.com/personal/qianshh_shanghaitech_edu_cn/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fqianshh%5Fshanghaitech%5Fedu%5Fcn%2FDocuments%2Fshare%2FProjects%2Fhrnet%2Dfor%2Dfashion%2Dlandmark%2Destimation%2Epytorch%2Fmodels)
   * source-git (HRNet-for-Fashion-Landmark-Estimation.PyTorch) : (https://github.com/svip-lab/HRNet-for-Fashion-Landmark-Estimation.PyTorch)

Next, run fashion ***landmark/Pytorch to TFLite.ipynb*** in sequence to get the TFLite model.

Finally, move the generated *TFLite model* to the *auto_clothing_measure folder*.

<br/>
<br/>

### Inference

- Save the model landmark prediction results as a json file.  &rightarrow; **estimated_kpt.json** is created

```
python rgb_to_json.py -ip samples/point_cloud_sample1/rgb.jpg -jn samples/point_cloud_sample1/estimated_kpt.json
```

* Save length(cm) by category and visualization images using the created json file. 
   * Length measurement methodology (version1)  &rightarrow; **result_kpt_v1.json** and **result_image_v1.png** is created

   ```
   python json_to_landmark_v1.py -r samples/point_cloud_sample1
   ```
  
  *  Length measurement methodology (version2)  &rightarrow; **new_ply.ply** , **result_kpt_v2.json** and **result_image_v2.png** is created

   ```
   python json_to_landmark_v2.py -r samples/point_cloud_sample1
   ```

<br/>

### Result

- Input Data
<img src="https://user-images.githubusercontent.com/37736774/215266341-b3602353-2717-44f4-b7e3-b001618a5e09.JPG" width="800" height="200"/>

- Actual Length
<img src="https://user-images.githubusercontent.com/37736774/215266652-d176cdb7-36bb-4be8-a807-47f091bb3121.JPG" width="600" height="200"/>

- Prediction - version1
<img src="https://user-images.githubusercontent.com/37736774/215266712-d25791dd-5786-4ec9-8054-7d925872fcc2.png" width="600" height="500"/>

- Prediction - version2
<img src="https://user-images.githubusercontent.com/37736774/215266732-a004b77e-44fc-4d68-b2ba-3ed59028f6b9.png" width="600" height="500"/>


<br/>
<br/>

### To Do

- Mobile net-based model training and testing
 
- Create more precise (enhanced) point cloud data from depth data

- Improving the performance of the final model (including the stable training phase)














