from typing import List, Tuple, Union

import cv2
import numpy as np
import PIL.Image as Image


class PrePostProcessor:
    def __init__(
        self,
        input_shape: Tuple[int] = (1, 3, 640, 640),
        input_type: np.dtype = np.float32,
        score_threshold: float = 0.35,
        nms_threshold: float = 0.45,
    ):
        self.input_shape = input_shape
        self.input_type = input_type
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold

    def preprocess(self, image: Union[str, np.ndarray], pad_color: Tuple = (114, 114, 114)) -> Tuple[np.ndarray, float]:
        """
        对检测任务的输入数据进行预处理:
            1. 对输入图像进行缩放
            2. 归一化

        参数:
            image (Union[Image.Image, np.ndarray]): 输入数据
            input_shape (Tuple[int]): 模型输入层所需的图像大小
            input_type (np.dtype): 输入数据类型，默认为np.float32
            pad_color (Tuple): 图像填充颜色，默认为(114, 114, 114)

        返回:
            Tuple[np.ndarray, float]: 预处理后的图像和缩放比例
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            image = Image.open(image)
        width, height = image.size
        scale = 1.0 / max(width / self.input_shape[2], height / self.input_shape[3])
        image = image.resize((round(width * scale), round(height * scale)), resample=Image.BILINEAR)
        pad = Image.new("RGB", (self.input_shape[2], self.input_shape[3]), pad_color)
        pad.paste(image)
        image = np.asarray(pad, dtype=self.input_type) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        return image, scale

    def postprocess(self, outputs: List[np.ndarray], scale: float) -> List:
        """
        后处理检测输出。

        参数:
            outputs(np.ndarray): 检测输出
            scale(float): 输入的缩放比例。

        返回:
            detections(List):后处理的检测结果, 格式为:[xmin, ymin, xmax, ymax, score, class_id]
        """
        # 将输出进行转置以匹配预期的形状
        outputs = np.transpose(outputs, (0, 2, 1))  # [1, 84, 8400] -> [1, 8400, 84]
        detections = []
        for batch in outputs:
            # 检测结果切片
            boxes, classes_scores = np.split(batch, [4], axis=1)
            class_ids = np.argmax(classes_scores, axis=1)
            scores = classes_scores[np.arange(len(class_ids)), class_ids]
            # 计算边界框的坐标 xywh -> xyxy
            boxes /= scale
            boxes[:, :2] -= boxes[:, 2:] / 2
            boxes[:, 2:] += boxes[:, :2]
            # NMS
            indices = cv2.dnn.NMSBoxesBatched(boxes, scores, class_ids, self.score_threshold, self.nms_threshold)
            if len(indices) > 0:
                detections.append(np.column_stack([boxes[indices], scores[indices], class_ids[indices]]))
            else:
                detections.append(np.empty((0, 6)))

        return detections
