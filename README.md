# YOLOv8.INFER.PY

用于基于 YoloV8 模型进行对象检测的工具。

## 项目概述

该项目是一个用于对象检测的工具，基于 YoloV8 模型。它提供了两种推理引擎的支持：OpenVINO 和 ONNX Runtime。你可以使用这个工具来对图像进行对象检测，识别其中的物体，并在图像中标注它们。

## 如何安装

```bash
pip install -r requirements.txt
```

## 如何运行

项目支持两种不同的推理引擎：OpenVINO 和 ONNX Runtime。你可以根据自己的需求选择一个引擎来运行对象检测。

具体参考[benchmark.py](benchmark.py)。

## 模型转换

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx,openvino
```

## 许可证

这个项目使用 [MIT 许可证](LICENSE)。
