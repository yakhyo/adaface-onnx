# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import numpy as np
from onnxruntime import InferenceSession

from utils import distance2bbox, distance2kps, get_onnx_providers

__all__ = ["SCRFD"]


class SCRFD:
    """SCRFD Face Detector.

    Title: "Sample and Computation Redistribution for Efficient Face Detection"
    Paper: https://arxiv.org/abs/2105.04714
    """

    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int] = (640, 640),
        conf_thres: float = 0.5,
        iou_thres: float = 0.4,
        providers: list[str] | None = None,
    ) -> None:
        """Initialize SCRFD face detector.

        Args:
            model_path: Path to ONNX model file.
            input_size: Model input size (width, height). Defaults to (640, 640).
            conf_thres: Confidence threshold. Defaults to 0.5.
            iou_thres: NMS IoU threshold. Defaults to 0.4.
            providers: ONNX Runtime execution providers.
                Defaults to ["CUDAExecutionProvider", "CPUExecutionProvider"].
        """
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.providers = providers or get_onnx_providers()

        # Model architecture parameters
        self.num_feature_maps = 3
        self.feat_strides = [8, 16, 32]
        self.num_anchors = 2
        self.use_keypoints = True

        # Normalization parameters
        self.mean = 127.5
        self.std = 128.0

        # Cache for anchor centers
        self.anchor_cache: dict[tuple[int, int, int], np.ndarray] = {}

        # Initialize ONNX session
        self.session = InferenceSession(model_path, providers=self.providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def _forward(self, image: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Run forward pass on preprocessed image.

        Args:
            image: Preprocessed input image.

        Returns:
            Tuple of (scores_list, bboxes_list, keypoints_list) for each feature map.
        """
        scores_list = []
        bboxes_list = []
        keypoints_list = []

        input_size = tuple(image.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(image, 1.0 / self.std, input_size, (self.mean, self.mean, self.mean), swapRB=True)
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]

        for idx, stride in enumerate(self.feat_strides):
            scores = outputs[idx]
            bbox_preds = outputs[idx + self.num_feature_maps] * stride

            if self.use_keypoints:
                kps_preds = outputs[idx + self.num_feature_maps * 2] * stride

            height = input_height // stride
            width = input_width // stride
            cache_key = (height, width, stride)

            if cache_key in self.anchor_cache:
                anchor_centers = self.anchor_cache[cache_key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self.num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self.num_anchors, axis=1).reshape((-1, 2))
                if len(self.anchor_cache) < 100:
                    self.anchor_cache[cache_key] = anchor_centers

            positive_indices = np.where(scores >= self.conf_thres)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)

            scores_list.append(scores[positive_indices])
            bboxes_list.append(bboxes[positive_indices])

            if self.use_keypoints:
                keypoints = distance2kps(anchor_centers, kps_preds)
                keypoints = keypoints.reshape((keypoints.shape[0], -1, 2))
                keypoints_list.append(keypoints[positive_indices])

        return scores_list, bboxes_list, keypoints_list

    def detect(
        self,
        image: np.ndarray,
        max_num: int = 0,
        metric: str = "max",
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Detect faces in an image.

        Args:
            image: Input image in BGR format.
            max_num: Maximum number of faces to return. 0 means no limit.
            metric: Selection metric when max_num > 0.
                "max" selects largest faces, "center" prefers centered faces.

        Returns:
            Tuple of (detections, keypoints):
                - detections: Array of shape (N, 5) with [x1, y1, x2, y2, score].
                - keypoints: Array of shape (N, 5, 2) with facial landmarks, or None.
        """
        target_width, target_height = self.input_size

        # Calculate resize ratio while preserving aspect ratio
        image_ratio = float(image.shape[0]) / image.shape[1]
        model_ratio = target_height / target_width

        if image_ratio > model_ratio:
            new_height = target_height
            new_width = int(new_height / image_ratio)
        else:
            new_width = target_width
            new_height = int(new_width * image_ratio)

        scale = float(new_height) / image.shape[0]
        resized_image = cv2.resize(image, (new_width, new_height))

        # Pad to target size
        padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        padded_image[:new_height, :new_width, :] = resized_image

        # Run detection
        scores_list, bboxes_list, keypoints_list = self._forward(padded_image)

        # Aggregate results from all feature maps
        scores = np.vstack(scores_list)
        bboxes = np.vstack(bboxes_list) / scale
        keypoints = np.vstack(keypoints_list) / scale if self.use_keypoints else None

        # Sort by score
        order = scores.ravel().argsort()[::-1]

        # Apply NMS
        detections = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
        detections = detections[order, :]
        keep_indices = self._nms(detections)
        detections = detections[keep_indices, :]

        if keypoints is not None:
            keypoints = keypoints[order, :, :]
            keypoints = keypoints[keep_indices, :, :]

        # Limit number of detections
        if 0 < max_num < detections.shape[0]:
            area = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])

            if metric == "max":
                values = area
            else:
                # Prefer faces closer to image center
                image_center = (image.shape[0] // 2, image.shape[1] // 2)
                offsets = np.vstack(
                    [
                        (detections[:, 0] + detections[:, 2]) / 2 - image_center[1],
                        (detections[:, 1] + detections[:, 3]) / 2 - image_center[0],
                    ]
                )
                offset_dist_squared = np.sum(np.power(offsets, 2.0), axis=0)
                values = area - offset_dist_squared * 2.0

            top_indices = np.argsort(values)[::-1][:max_num]
            detections = detections[top_indices, :]
            if keypoints is not None:
                keypoints = keypoints[top_indices, :]

        return detections, keypoints

    def _nms(self, detections: np.ndarray) -> list[int]:
        """Apply Non-Maximum Suppression.

        Args:
            detections: Array of shape (N, 5) with [x1, y1, x2, y2, score].

        Returns:
            List of indices to keep.
        """
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        scores = detections[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)

            remaining = np.where(iou <= self.iou_thres)[0]
            order = order[remaining + 1]

        return keep
