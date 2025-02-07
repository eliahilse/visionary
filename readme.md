# Visionary

## Setup

### OpenCV
The package manager [vcpkg](https://vcpkg.io/en/package/opencv4) was utilized to install OpenCV (4.10).

Installation:\
! Ensure that the environmental variable `VCPKG_ROOT` points to your vcpkg installation for Cmake to pick it up properly.
- CPU -> `vcpkg install opencv4[dnn]`
- GPU, with CUDA support -> `vcpkg install opencv4[dnn-cuda]`\
  *Note: you may need to separately install CUDA and CuDNN from Nvidia, if not present on your system.*
- GPU, without CUDA support -> `vcpkg install opencv4[dnn,opencl]`\
  *Note: OPENCV_OCL4DNN_CONFIG_PATH tba*

You can also explicitly tell vcpkg which platform to build for by adding the postfix `:your-platform` to the end of the command.
Regarding vcpkg's supported platforms, refer to [here](https://github.com/microsoft/vcpkg/tree/master/triplets).

Me, working on Windows 64-Bit with CUDA, used `vcpkg install opencv4[dnn-cuda]`

Pro tip:
For sped up builds, we recommend setting the `CMAKE_BUILD_PARALLEL_LEVEL` environmental variable to the amount of logical processors (threads) your CPU supports, to ensure maximum parallelization.
On Windows, that can be done with the command `set CMAKE_BUILD_PARALLEL_LEVEL={amount_max_threads}`.

### Yolo

#### YoloV7

The YoloV7 repository is included in the /yolo/v7 folder, as a git submodule.
To have it output a model in the onnx (dnn runtime compatible) file format, run the following command:
`python export.py --weights yolov7-tiny.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640`

#### YoloV9

The YoloV9 repository is included in the /yolo/v9 folder, as a git submodule.

To build using it, you either need to download a pretrained model (e.g. via the ones pretrained on the COCO set via https://github.com/WongKinYiu/yolov9/?tab=readme-ov-file#performance)
s
To then convert the .pt file to .onnx via the ONNX pipeline:
`python export.py --weights yolov9-m.pt --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --imgsz 640 640 --include onnx`
(or `python3` for unix systems)

### OC-Sort

The OC-Sort repository is included in the /oc-sort folder, as a git submodule.

## Samples
<div style="">
  <img src="docs/assets/sample-1.jpg" alt="Sample Image" />
  <p style="font-size: smaller; margin-top: 0px;">Yolo v9m, OC-Sort & Cross-Camera matching: 43ms inference on 2x 640x640 camera input feeds</p>
</div>

## Attributions
Wong Kin-Yiu, for his [YoloV7](https://github.com/WongKinYiu) and [YoloV9](https://github.com/WongKinYiu/yolov9) implementations.
Jinkun Cao for [OC-Sort](https://github.com/noahcao/OC_SORT).\
Fernando B. Giannasi for his [implementation](https://github.com/phoemur/hungarian_algorithm) of the Hungarian algorithm in C++.

