#!/bin/bash

# Download AdaFace ONNX model weights from GitHub releases
# Usage: ./download.sh

BASE_URL="https://github.com/yakhyo/adaface-onnx/releases/download/weights"

WEIGHTS_DIR="weights"
mkdir -p ${WEIGHTS_DIR}

echo "Downloading model weights..."

# Face detector
echo "Downloading SCRFD detector (det_10g.onnx)..."
curl -L "${BASE_URL}/det_10g.onnx" -o "${WEIGHTS_DIR}/det_10g.onnx"

# AdaFace models
echo "Downloading AdaFace IR-18 (adaface_ir_18.onnx)..."
curl -L "${BASE_URL}/adaface_ir_18.onnx" -o "${WEIGHTS_DIR}/adaface_ir_18.onnx"

echo "Downloading AdaFace IR-101 (adaface_ir_101.onnx)..."
curl -L "${BASE_URL}/adaface_ir_101.onnx" -o "${WEIGHTS_DIR}/adaface_ir_101.onnx"

echo "Done! Weights saved to ${WEIGHTS_DIR}/"
ls -lh ${WEIGHTS_DIR}/

