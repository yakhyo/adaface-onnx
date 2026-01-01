# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import numpy as np
from onnxruntime import InferenceSession

from utils import face_alignment, get_onnx_providers


class AdaFace:
    """AdaFace Model for Face Recognition

    This class implements a face encoder using the AdaFace architecture,
    loading a pre-trained model from an ONNX file.
    """

    def __init__(self, model_path: str, providers: list[str] | None = None) -> None:
        """Initializes the AdaFace face encoder model.

        Args:
            model_path: Path to ONNX model file.
            providers: List of ONNX Runtime execution providers.
                Defaults to ["CUDAExecutionProvider", "CPUExecutionProvider"].

        Raises:
            RuntimeError: If model initialization fails.
        """
        self.model_path = model_path
        self.providers = providers or get_onnx_providers()
        self.input_size = (112, 112)
        self.normalization_mean = 127.5
        self.normalization_std = 127.5

        self.session = InferenceSession(self.model_path, providers=self.providers)

        input_config = self.session.get_inputs()[0]
        self.input_name = input_config.name
        input_shape = input_config.shape
        model_input_size = tuple(input_shape[2:4][::-1])

        if model_input_size != self.input_size:
            print(f"Warning: Model input size {model_input_size} differs from configured size {self.input_size}")

        self.output_names = [o.name for o in self.session.get_outputs()]
        self.output_shape = self.session.get_outputs()[0].shape
        self.embedding_size = self.output_shape[1]

        assert len(self.output_names) == 1, "Expected only one output node."

    def preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess the face image: resize, normalize, and convert to the required format.

        Args:
            face_image: Input face image in BGR format.

        Returns:
            Preprocessed image blob ready for inference.
        """
        resized_face = cv2.resize(face_image, self.input_size)

        if isinstance(self.normalization_std, (list, tuple)):
            # Handle per-channel normalization (keep BGR)
            mean_array = np.array(self.normalization_mean, dtype=np.float32)
            std_array = np.array(self.normalization_std, dtype=np.float32)
            normalized_face = (resized_face.astype(np.float32) - mean_array) / std_array

            # Change to NCHW format (batch, channels, height, width)
            face_blob = np.expand_dims(np.transpose(normalized_face, (2, 0, 1)), axis=0)
        else:
            # Single-value normalization using cv2.dnn (keep BGR)
            face_blob = cv2.dnn.blobFromImage(
                resized_face,
                scalefactor=1.0 / self.normalization_std,
                size=self.input_size,
                mean=(self.normalization_mean,) * 3,
                swapRB=False,
            )

        return face_blob

    def get_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Extract face embedding from an image using facial landmarks for alignment.

        Args:
            image: Input image in BGR format.
            landmarks: 5-point facial landmarks for alignment.

        Returns:
            Face embedding vector.

        Raises:
            ValueError: If inputs are invalid.
        """
        if image is None or landmarks is None:
            raise ValueError("Image and landmarks must not be None")

        aligned_face, _ = face_alignment(image, landmarks)
        face_blob = self.preprocess(aligned_face)
        embedding = self.session.run(self.output_names, {self.input_name: face_blob})[0]
        return embedding.flatten()

    def get_normalized_embedding(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Extract an L2-normalized face embedding vector from an image.

        Args:
            image: Input face image in BGR format.
            landmarks: Facial landmarks (5 points for alignment).

        Returns:
            L2-normalized face embedding vector (typically 512-dimensional).
        """
        embedding = self.get_embedding(image, landmarks)
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding
