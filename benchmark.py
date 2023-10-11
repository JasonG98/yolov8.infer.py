from PIL import Image
import time

from yolov8_infer import OnnxDetector, OpenvinoDetector


def benchmark():
    onnx_detector = OnnxDetector(
        model_path="models/yolov8s.onnx",
        score_threshold=0.35,
        nms_threshold=0.45,
    )

    openvino_detector = OpenvinoDetector(
        model_path="models/yolov8s/yolov8s.xml",
        score_threshold=0.35,
        nms_threshold=0.45,
        device="CPU",
    )

    image = Image.open("images/bus.jpg")

    start = time.time()
    for i in range(100):
        onnx_detector(image)
    end = time.time()
    print(f"onnx: {(end - start) / 100}")

    start = time.time()
    for i in range(100):
        openvino_detector(image)
    end = time.time()
    print(f"openvino: {(end - start) / 100}")


if __name__ == "__main__":
    benchmark()
