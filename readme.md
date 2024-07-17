# Visionary

Experiments for object detection in real-time, using YOLO v8.

## Pretrained Models
Pretrained Yolo models can be found e.g. [here](https://github.com/ultralytics/ultralytics).
Since most modern Yolo models are implemented in PyTorch, the usual format you find pretrained models usually is .pt.
With ultralytics python package, you can easily convert these .pt models to e.g. OpenCV compatible .onnx format.
```python
from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.export(format="onnx", opset=12, imgsz=[640, 640])
```
The ultralytics python package will also automatically attempt to download a suitable .pt file if not found locally.

## C++ Setup
For C++, OpenCV is required.\
Easiest way to install that is by utilizing vcpkg package manager.\
Make sure to install the CUDA version supposing you have an Nvidia GPU so that the model can run on the GPU instead of the CPU.\
Visual Studio 2022 project files are included. CMake file is on the way.

## Python Setup
Python can be ran either with OpenCV or Ultralytics.
In both cases, it's sufficient to `pip install -r requirements.txt`

## Samples
<div style="">
  <img src="docs/assets/sample-1.jpg" alt="Sample Image" />
  <p style="font-size: smaller; margin-top: 0px;">Yolo v8 XL, 30ms inference result on RTX 3070 Ti Laptop, Laptop Camera Feed (real-time)</p>
</div>
