# Meter Reader Pipeline

A comprehensive pipeline for automated meter reading using computer vision and deep learning. This project implements a three-stage pipeline for detecting meter dials, segmenting digits, and classifying the segmented digits.

## Project Structure

```
meter-reader-pipeline/
├── README.md
├── LICENSE
├── requirements.txt
├── config/                 # Configuration files for training
├── data/                  # Dataset organization
├── models/                # Model architectures
├── scripts/               # Training and evaluation scripts
├── utils/                 # Utility functions
└── checkpoints/           # Model checkpoints
```

## Features

- Dial Detection: YOLO-based model for detecting meter dials in images
- Digit Segmentation: U-Net based model for segmenting individual digits
- Digit Classification: CNN model for classifying segmented digits
- Comprehensive evaluation and ablation studies
- Robustness testing against various conditions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/meter-reader-pipeline.git
cd meter-reader-pipeline
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in the `data/raw` directory
2. Configure training parameters in the respective YAML files in `config/`
3. Train the models:
```bash
python scripts/train_dial.py
python scripts/train_digit_seg.py
python scripts/train_digit_cls.py
```

4. Evaluate the pipeline:
```bash
python scripts/evaluate_pipeline.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
