# Auto-measure-clothing-dimensions
Deep learning based automatic clothing length measurement project using PointCloud and RGB images

## purpose of the project
Using RGB images and Point Cloud data, build a deep learning-based automatic clothing length measurement system that can operate on-device inside a mobile device.

## Overview
<img src="https://user-images.githubusercontent.com/37736774/215036841-c9c5aad5-bcf0-4693-a067-b5d56d18f0cb.png" width="800" height="400"/>

   
   
<br/>
<br/>


## code process

#### preparation 

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





