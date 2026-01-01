# Copyright 2026 Yakhyokhuja Valikhujaev
# Author: Yakhyokhuja Valikhujaev
# GitHub: https://github.com/yakhyo

import cv2
import numpy as np
import onnxruntime
from skimage.transform import SimilarityTransform

__all__ = [
    "compute_similarity",
    "face_alignment",
    "get_onnx_providers",
]


def get_onnx_providers() -> list[str]:
    """Get available ONNX Runtime execution providers.

    Returns providers in order of preference: CUDA > CoreML > CPU.

    Returns:
        List of available provider names.
    """
    available = onnxruntime.get_available_providers()

    # Preferred order
    preferred = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]

    return [p for p in preferred if p in available]


# Standard 5-point facial landmark reference for ArcFace alignment (112x112)
reference_alignment: np.ndarray = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def estimate_norm(
    landmark: np.ndarray,
    image_size: int | tuple[int, int] = 112,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the normalization transformation matrix for facial landmarks.

    Args:
        landmark: Array of shape (5, 2) representing the coordinates of the facial landmarks.
        image_size: The size of the output image. Can be an integer (for square images)
            or a tuple (width, height). Default is 112.

    Returns:
        A tuple containing:
            - The 2x3 transformation matrix for aligning the landmarks.
            - The 2x3 inverse transformation matrix.

    Raises:
        AssertionError: If the input landmark array does not have the shape (5, 2)
            or if image_size is not a multiple of 112 or 128.
    """
    assert landmark.shape == (5, 2), "Landmark array must have shape (5, 2)."

    # Handle both int and tuple inputs
    if isinstance(image_size, tuple):
        size = image_size[0]  # Use width for ratio calculation
    else:
        size = image_size

    assert size % 112 == 0 or size % 128 == 0, "Image size must be a multiple of 112 or 128."

    if size % 112 == 0:
        ratio = float(size) / 112.0
        diff_x = 0.0
    else:
        ratio = float(size) / 128.0
        diff_x = 8.0 * ratio

    # Adjust reference alignment based on ratio and diff_x
    alignment = reference_alignment * ratio
    alignment[:, 0] += diff_x

    # Compute the transformation matrix
    transform = SimilarityTransform()
    transform.estimate(landmark, alignment)

    matrix = transform.params[0:2, :]
    inverse_matrix = np.linalg.inv(transform.params)[0:2, :]

    return matrix, inverse_matrix


def face_alignment(
    image: np.ndarray,
    landmark: np.ndarray,
    image_size: int | tuple[int, int] = 112,
) -> tuple[np.ndarray, np.ndarray]:
    """Align the face in the input image based on the given facial landmarks.

    Args:
        image: Input image as a NumPy array with shape (H, W, C).
        landmark: Array of shape (5, 2) representing the facial landmark coordinates.
        image_size: The size of the aligned output image. Can be an integer
            (for square images) or a tuple (width, height). Default is 112.

    Returns:
        A tuple containing:
            - The aligned face as a NumPy array.
            - The 2x3 inverse transformation matrix used for alignment.
    """
    # Get the transformation matrix
    transform_matrix, inverse_transform = estimate_norm(landmark, image_size)

    # Handle both int and tuple for warpAffine output size
    if isinstance(image_size, int):
        output_size = (image_size, image_size)
    else:
        output_size = image_size

    # Warp the input image to align the face
    warped = cv2.warpAffine(image, transform_matrix, output_size, borderValue=0.0)

    return warped, inverse_transform


def compute_similarity(feat1: np.ndarray, feat2: np.ndarray, normalized: bool = False) -> np.float32:
    """Compute cosine similarity between two face embeddings.

    Args:
        feat1: First embedding vector.
        feat2: Second embedding vector.
        normalized: Set True if the embeddings are already L2 normalized.

    Returns:
        Cosine similarity score in range [-1, 1].
    """
    feat1 = feat1.ravel()
    feat2 = feat2.ravel()
    if normalized:
        return np.dot(feat1, feat2)
    # Add small epsilon to prevent division by zero
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2) + 1e-5)


def distance2bbox(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Decode distance prediction to bounding box.

    Args:
        points: Shape (n, 2), anchor center [x, y].
        distance: Distance from the anchor to 4 boundaries (left, top, right, bottom).
        max_shape: Shape of the image (height, width) for clipping.

    Returns:
        Decoded bounding boxes with shape (n, 4).
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(
    points: np.ndarray,
    distance: np.ndarray,
    max_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Decode distance prediction to keypoints.

    Args:
        points: Shape (n, 2), anchor center [x, y].
        distance: Distance predictions for keypoints.
        max_shape: Shape of the image (height, width) for clipping.

    Returns:
        Decoded keypoints with shape (n, 2k).
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)
