from typing import List, Union

import numpy as np
import onnxruntime as ort

from .prepostprocessor import PrePostProcessor


class Detector:
    """ONNXRuntime 模型"""

    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.35,
        nms_threshold: float = 0.45,
    ):
        self.session = ort.InferenceSession(model_path, providers=["AzureExecutionProvider", "CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        input_shape = self.session.get_inputs()[0].shape
        input_type = np.float32

        self.ppp = PrePostProcessor(input_shape, input_type, score_threshold, nms_threshold)

    def __call__(self, image: Union[str, np.ndarray]) -> List[np.ndarray]:
        return self.predict(image)

    def predict(self, image: Union[str, np.ndarray]) -> List[np.ndarray]:
        """
        进行推理
        """
        blob, scale = self.ppp.preprocess(image)
        outputs = self.session.run(None, {self.input_name: blob})[0]
        return self.ppp.postprocess(outputs, scale)
