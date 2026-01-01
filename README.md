# AdaFace ONNX

[![Downloads](https://img.shields.io/github/downloads/yakhyo/adaface-onnx/total.svg)](https://github.com/yakhyo/adaface-onnx/releases)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-yakhyo-black.svg?logo=github)](https://github.com/yakhyo)

ONNX Runtime inference for [AdaFace](https://github.com/mk-minchul/AdaFace) face recognition with [SCRFD](https://github.com/deepinsight/insightface) face detection.

## Models

ONNX weights exported from official PyTorch weights. Download via `bash download.sh` or from [Releases](https://github.com/yakhyo/adaface-onnx/releases/tag/weights).

| Model  | Dataset    | IJBB TAR@FAR=0.01% | IJBC TAR@FAR=0.01% | Size   | Download |
| ------ | ---------- | ------------------ | ------------------ | ------ | -------- |
| IR-18  | WebFace4M  | 93.03              | 94.99              | 92 MB  | [Link](https://github.com/yakhyo/adaface-onnx/releases/download/weights/adaface_ir_18.onnx) |
| IR-101 | WebFace12M | -                  | 97.66              | 249 MB | [Link](https://github.com/yakhyo/adaface-onnx/releases/download/weights/adaface_ir_101.onnx) |

Face detector: **SCRFD-10G** (16 MB) - [Download](https://github.com/yakhyo/adaface-onnx/releases/download/weights/det_10g.onnx)

## Installation

```bash
pip install -r requirements.txt
bash download.sh  # Download model weights
```

## Demo

### Aligned Faces (112x112)

|                   Anthony Edward                   |                   Anthony Edward                   |                   Nicolas Cage                   |
| :------------------------------------------------: | :------------------------------------------------: | :----------------------------------------------: |
| ![Anthony Edward](assets/aligned/img1_aligned.jpg) | ![Anthony Edward](assets/aligned/img2_aligned.jpg) | ![Nicolas Cage](assets/aligned/img3_aligned.jpg) |

### Similarity Matrix

|     | p1    | p2    | p3    |
| --- | ----- | ----- | ----- |
| p1  | 1.00  | 0.64  | -0.04 |
| p2  | 0.64  | 1.00  | -0.01 |
| p3  | -0.04 | -0.01 | 1.00  |

> p1 and p2 are the same person (similarity: **0.64**), p3 is different.

## Usage

### CLI

```bash
python main.py image1.jpg image2.jpg
```

### Python API

```python
from model import FaceRecognition

fr = FaceRecognition()
similarity = fr.compare("image1.jpg", "image2.jpg")
is_same = fr.is_match("image1.jpg", "image2.jpg", threshold=0.4)
```

## Reference

- [AdaFace](https://github.com/mk-minchul/AdaFace) - Original PyTorch implementation
- [SCRFD](https://github.com/deepinsight/insightface/tree/master/detection/scrfd) - Face detection

## License

MIT License
