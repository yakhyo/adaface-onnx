# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import numpy as np

from utils import compute_similarity

from .adaface import AdaFace
from .scrfd import SCRFD


class FaceRecognition:
    """High-level face recognition API combining SCRFD detector and AdaFace encoder."""

    def __init__(
        self,
        detector_path: str = "weights/det_10g.onnx",
        recognition_path: str = "weights/adaface_ir_18.onnx",
    ) -> None:
        self.detector = SCRFD(model_path=detector_path)
        self.recognizer = AdaFace(model_path=recognition_path)

    def get_embedding(self, image: np.ndarray) -> np.ndarray | None:
        """Extract face embedding from image. Returns None if no face detected."""
        detections, keypoints = self.detector.detect(image)
        if len(detections) == 0:
            return None
        return self.recognizer.get_normalized_embedding(image, keypoints[0])

    def compare(self, image1: np.ndarray | str, image2: np.ndarray | str) -> float | None:
        """Compare two face images. Returns similarity score or None if detection fails."""
        if isinstance(image1, str):
            image1 = cv2.imread(image1)
        if isinstance(image2, str):
            image2 = cv2.imread(image2)

        emb1 = self.get_embedding(image1)
        emb2 = self.get_embedding(image2)

        if emb1 is None or emb2 is None:
            return None

        return compute_similarity(emb1, emb2, normalized=True)

    def is_match(
        self,
        image1: np.ndarray | str,
        image2: np.ndarray | str,
        threshold: float = 0.4,
    ) -> bool | None:
        """Check if two images contain the same person."""
        similarity = self.compare(image1, image2)
        if similarity is None:
            return None
        return similarity >= threshold
