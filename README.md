# Autodistill YOLOE

[![PyPI - Version](https://img.shields.io/pypi/v/autodistill-yoloe.svg)](https://pypi.org/project/autodistill-yoloe)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/autodistill-yoloe.svg)](https://pypi.org/project/autodistill-yoloe)

## Description

Use YOLOE to auto-label images for use in training fine-tuned object detection models

## Overview

Autodistill YOLOE is a package that allows you to use [YOLOE](https://github.com/ultralytics/ultralytics) models with the [Autodistill](https://github.com/autodistill/autodistill) framework. YOLOE is a state-of-the-art object detection model that supports both text/visual prompting and prompt-free detection.

This integration enables you to:

1. Use YOLOE models to automatically label images
2. Prepare datasets for training custom object detection models
3. Leverage both text-prompted and prompt-free detection capabilities

## Installation

Python 3.11 or later is required

```bash
pip install autodistill-yoloe
```

## Usage

### Basic Example

```python
from autodistill_yoloe import YOLOEBase
from autodistill.detection import CaptionOntology

# Define your ontology (the objects you want to detect)
ontology = CaptionOntology({"person": "person", "car": "car", "dog": "dog"})

# Initialize the YOLOE model
yoloe = YOLOEBase(ontology=ontology)

# Label a folder of images
dataset = yoloe.label("path/to/images", extension=".jpg")

# The labeled dataset is ready for training a custom model
print("Dataset created and ready for distillation!")
```

### Using a Custom YOLOE Model

```python
from autodistill_yoloe import YOLOEBase
from autodistill.detection import CaptionOntology

ontology = CaptionOntology({"person": "person", "car": "car"})

# Use a specific YOLOE model
yoloe = YOLOEBase(ontology=ontology, model="yoloe-11s-seg.pt")

# Make predictions on a single image
detections = yoloe.predict("path/to/image.jpg")
print(detections)
```

### Human-in-the-Loop Labeling with Roboflow

```python
from autodistill_yoloe import YOLOEBase
from autodistill.detection import CaptionOntology

ontology = CaptionOntology({"person": "person", "car": "car"})
yoloe = YOLOEBase(ontology=ontology)

# Enable human-in-the-loop labeling with Roboflow
dataset = yoloe.label(
    "path/to/images",
    human_in_the_loop=True,
    roboflow_project="my-project-name"
)
```

## Available Models

YOLOE supports several model variants:

### Text/Visual Prompt Models

- YOLOE-11S
- YOLOE-11M
- YOLOE-11L
- YOLOE-v8S
- YOLOE-v8M
- YOLOE-v8L

### Prompt-Free Models

- YOLOE-11S-PF
- YOLOE-11M-PF
- YOLOE-11L-PF
- YOLOE-v8S-PF
- YOLOE-v8M-PF
- YOLOE-v8L-PF

## Advanced Features

- **SAHI Integration**: Enable SAHI (Slicing Aided Hyper Inference) for improved detection of small objects
- **NMS Settings**: Configure Non-Maximum Suppression for better detection results
- **Confidence Recording**: Save confidence scores with detections

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [Autodistill](https://github.com/autodistill/autodistill) - The framework that makes this integration possible
- [Ultralytics](https://github.com/ultralytics/ultralytics) - Creators of the YOLOE model
- [Supervision](https://github.com/roboflow/supervision) - Computer vision toolkit used for annotations
