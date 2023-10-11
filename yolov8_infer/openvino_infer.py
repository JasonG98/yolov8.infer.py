from typing import List, Union

import numpy as np
import openvino as ov

from .prepostprocessor import PrePostProcessor


class Detector:
    """OpenVINO模型"""

    def __init__(
        self,
        model_path: str,
        score_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        device: str = "CPU",
    ):
        core = ov.Core()
        core.set_property({"CACHE_DIR": "./cache"})
        compiled_model = core.compile_model(model_path, device)

        self.ir = compiled_model.create_infer_request()
        self.output_node = compiled_model.outputs[0]

        input_shape = self.ir.get_input_tensor(0).data.shape
        input_type = self.ir.get_input_tensor(0).data.dtype

        self.ppp = PrePostProcessor(input_shape, input_type, score_threshold, nms_threshold)

    def __call__(self, image: Union[str, np.ndarray]) -> List[np.ndarray]:
        return self.predict(image)

    def predict(self, image: Union[str, np.ndarray]) -> List[np.ndarray]:
        """
        进行推理
        """
        blob, scale = self.ppp.preprocess(image)
        outputs = self.ir.infer(blob)[self.output_node]
        return self.ppp.postprocess(outputs, scale)
