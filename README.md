# Sidewalk Hazard Detection Using VAE and One-Class SVM

[![Dataset](https://img.shields.io/badge/Dataset-SONE-green)](https://doi.org/10.7910/DVN/ZLYKI9)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of "Sidewalk Hazard Detection Using a Variational Autoencoder and One-Class SVM"

## Overview

This repository contains the code for a hybrid anomaly detection system that combines a Variational Autoencoder (VAE) with a One-Class Support Vector Machine (OCSVM) to detect hazardous obstacles on sidewalks. The system achieves:

- **AUC**: 0.92 (average across conditions)
- **F1 Score**: 0.85
- **Real-time performance**: 39.29 ms per frame on consumer hardware

### Key Features

- **Unsupervised learning**: Trained only on normal sidewalk images
- **Two-stage detection**: VAE identifies anomalies, OCSVM classifies hazard severity
- **Robust performance**: Tested across multiple locations and weather conditions
- **Lightweight**: Runs in real-time on consumer laptops
- **Egocentric dataset**: Novel SONE dataset with 20,000+ training and 8,000+ testing frames

## System Architecture

```
Input RGB Frame → VAE (Anomaly Detection) → OCSVM (Hazard Classification) → Alert User
                    ↓
              Low reconstruction → Anomaly present
                    ↓
              OCSVM Decision:
                +1: Non-hazardous (traversable)
                -1: Hazardous (non-traversable)
```

## Installation

### Requirements

```bash
# Python 3.8+
pip install torch torchvision
pip install scikit-learn
pip install numpy
pip install tqdm
pip install wandb  # For experiment tracking
```

### Clone Repository

```bash
git clone https://github.com/erguzman0808/Sidewalk-Hazard-Detection-Using-a-Variational-Autoencoder-and-One-Class-SVM.git
cd sidewalk-hazard-detection
```

## Dataset

The **SONE (Sidewalk Outdoor Navigation Environment)** dataset is available at:  
**[https://doi.org/10.7910/DVN/ZLYKI9](https://doi.org/10.7910/DVN/ZLYKI9)**

### Dataset Structure

```
Data/
├── output_images/
│   └── Outdoor_Training/
│       └── color/           # 20,000+ normal sidewalk frames
└── testing/
    ├── hazardous/           # Frames with hazardous anomalies
    └── non_hazardous/       # Frames with traversable anomalies
```

### Data Collection Details

- **Camera**: Intel RealSense D435i
- **Resolution**: 640×480 RGB
- **FOV**: 87° × 58°
- **Frame Rate**: 30 fps
- **Camera Position**: Chest height, tilted 60° downward
- **Locations**: Massachusetts, California, Mexico City
- **Conditions**: Sunny, cloudy, various lighting and weather

## Usage

### 1. Train the VAE

Train the Variational Autoencoder on normal sidewalk images:

```bash
python VAE_Training.py \
  --data-path /path/to/training/data \
  --latent-dim 1024 \
  --batch-size 25 \
  --num-epochs 120 \
  --learning-rate 1e-5 \
  --kld-weight 1.0 \
  --output-dir ./trained_models
```

### 2. Train the OCSVM

Train the One-Class SVM on non-hazardous anomalies:

```bash
python OCSVM.py \
  --vae-model ./trained_models/vae_best.pth \
  --training-data /path/to/non_hazardous/anomalies \
  --output-dir ./ocsvm_models
```

The OCSVM uses:
- **Kernel**: RBF (Radial Basis Function)
- **Nu**: 0.001 (allows 0.1% outliers)
- **Gamma**: Scale

### 3. Evaluate Performance

Generate ROC curves and performance metrics:

```bash
python ROC_Curve_Results_svm.py \
  --vae-model ./trained_models/vae_best.pth \
  --ocsvm-model ./ocsvm_models/ocsvm_best.pkl \
  --test-data /path/to/test/data \
  --output-dir ./results
```

This script generates:
- ROC curves for different sidewalk types
- Precision-Recall curves
- F1 scores, AUC metrics
- Inference time analysis


## Citation

If you use this code or the SONE dataset in your research, please cite:

## Dataset Citation

```bibtex
@data{DVN/ZLYKI9_2024,
  author = {Anonymous Author(s)},
  publisher = {Harvard Dataverse},
  title = {{SONE: Sidewalk Outdoor Navigation Environment Dataset}},
  year = {2024},
  version = {V1},
  doi = {10.7910/DVN/ZLYKI9},
  url = {https://doi.org/10.7910/DVN/ZLYKI9}
}
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contact

For questions or issues, please open an issue on GitHub or contact the authors.

---
