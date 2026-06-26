# 👁️ EyeCheck AI — Diabetic Retinopathy Screening System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=flat-square&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![Accuracy](https://img.shields.io/badge/Accuracy-87.7%25-green?style=flat-square)
![AUC](https://img.shields.io/badge/Mean%20AUC-0.963-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

> **AI-powered fundus image analysis for early diabetic retinopathy detection — built for low-cost, offline deployment in rural healthcare settings.**

---

## 🔍 Overview

EyeCheck AI is a hybrid ensemble deep learning system that analyzes retinal fundus photographs and classifies them into three diabetic retinopathy (DR) grades:

| Grade | Description | Action |
|---|---|---|
| 🟢 **No DR** | No signs of retinopathy | Routine annual screening |
| 🟡 **Mild DR** | Microaneurysms detected | Ophthalmology review in 6–12 months |
| 🔴 **Moderate DR** | Hemorrhages / exudates present | Urgent referral within 1–3 months |

---

## ✨ Key Features

- 🔬 **Hybrid Ensemble** — ResNet50 + EfficientNet-B4 predictions averaged for robust classification
- 🗺️ **GradCAM Explainability** — Heatmap overlay showing exactly which retinal regions triggered the AI decision
- 🛡️ **Image Quality Gating** — Detects and rejects blurry, dark, or low-contrast images before analysis
- ⚖️ **Class Imbalance Handling** — WeightedRandomSampler + Weighted CrossEntropyLoss for balanced minority class detection
- 💻 **Offline Ready** — Runs entirely on a local laptop, no internet or cloud subscription required
- 🏥 **Clinical UI** — Urgency level, per-class confidence scores, per-model breakdown, and medical action recommendation

---

## 📊 Model Performance

### Validation Accuracy (APTOS 2019 Dataset)

| Model | Overall Accuracy |
|---|---|
| ResNet50 | **87.7%** |
| EfficientNet-B4 | **85.2%** |
| **Ensemble (avg)** | **~86.5%** |

### Per-Class Accuracy (ResNet50 Best Model)

| Class | Accuracy |
|---|---|
| No DR | 96.5% |
| Mild DR | 87.5% |
| Moderate DR | 73.4% |

### ROC AUC Scores (Ensemble)

| Class | AUC Score |
|---|---|
| No DR | **0.994** |
| Moderate DR | **0.968** |
| Mild DR | **0.928** |
| **Mean AUC** | **0.963** |

### Comparison with Existing Systems

| System | Accuracy | Training Data |
|---|---|---|
| Google DeepMind (2016) | 90.3% | 128,000 images |
| IDx-DR (FDA Approved) | 87.4% | Proprietary |
| **EyeCheck AI** | **87.7%** | **2,930 images** |

> ✦ Within 3% of Google DeepMind's accuracy — using 44× less data, on a CPU laptop.
> ✦ Mean AUC of 0.963 — Google's 2016 paper reported 0.991 with 44× more data.

---

## 📈 Evaluation Results

### Training Curves — ResNet50
![ResNet50 Training Curves](evaluation/01_resnet50_curves.png)

### Training Curves — EfficientNet-B4
![EfficientNet Training Curves](evaluation/02_efficientnet_curves.png)

### Confusion Matrix — Ensemble
![Confusion Matrix](evaluation/03_confusion_matrix.png)

> No DR: **99.4%** correct · Mild DR: **75.0%** correct · Moderate DR: **78.6%** correct

### ROC Curve — Ensemble (One-vs-Rest)
![ROC Curve](evaluation/04_roc_curve.png)

> Mean AUC = **0.963** — well above the 0.9 clinical threshold for diagnostic tools.

---

## 🧠 System Architecture

```
Fundus Image Upload
        │
        ▼
Image Quality Check (Sharpness · Brightness · Contrast)
        │
   Poor Quality → ❌ Reject with warning
        │
   Good Quality → ✅ Continue
        │
        ▼
Preprocessing (Resize 224×224 · Normalize RGB)
        │
        ├─────────────────────────┐
        ▼                         ▼
   ResNet50                 EfficientNet-B4
   (layer3+4+fc unfrozen)   (last 3 blocks unfrozen)
   Val Acc: 87.7%           Val Acc: 85.2%
        │                         │
        └──────────┬──────────────┘
                   ▼
         Ensemble Average (50% + 50%)
                   │
                   ▼
         Final Grade + Confidence Scores
                   │
                   ▼
         GradCAM Heatmap (ResNet50 layer4)
                   │
                   ▼
         Streamlit UI — Result + Medical Action
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch 2.0 |
| Models | ResNet50 + EfficientNet-B4 (torchvision) |
| Explainability | GradCAM (custom implementation) |
| Web App | Streamlit |
| Image Processing | OpenCV, PIL |
| Evaluation | scikit-learn (ROC, Confusion Matrix) |
| Dataset | APTOS 2019 Blindness Detection (Kaggle) |

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.11+
PyTorch 2.0+
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/EyeCheck-AI.git
cd EyeCheck-AI

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup
1. Download the APTOS 2019 dataset from [Kaggle](https://www.kaggle.com/c/aptos2019-blindness-detection)
2. Place files in this structure:
```
EyeCheck-AI/
├── archive(2)/
│   ├── train_images/train_images/
│   ├── val_images/val_images/
│   ├── train_1.csv
│   └── valid.csv
```

### Train the Models
```bash
python train_model.py
```
Training time: ~60–80 mins on CPU | ~15–20 mins on GPU

Expected output:
```
ResNet50       Best Val Acc : 0.877  (87.7%)
EfficientNet   Best Val Acc : 0.852  (85.2%)
```

### Launch the App
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`

### Generate Evaluation Plots
```bash
python evaluate.py
```
Saves confusion matrix, ROC curves, and accuracy/loss plots to `evaluation/`

---

## 📁 Project Structure

```
EyeCheck-AI/
├── app.py                          # Streamlit web application
├── train_model.py                  # Dual model training script
├── evaluate.py                     # Evaluation metrics and plots
├── requirements.txt                # Python dependencies
├── model/                          # Saved model weights (after training)
│   ├── resnet50_dr.pth
│   └── efficientnet_dr.pth
└── evaluation/                     # Generated plots (after evaluate.py)
    ├── 01_resnet50_curves.png
    ├── 02_efficientnet_curves.png
    ├── 03_confusion_matrix.png
    └── 04_roc_curve.png
```

---

## 🏥 Clinical Disclaimer

> This tool is an AI-assisted screening aid only — **not a substitute for professional clinical diagnosis**. All findings must be reviewed and confirmed by a certified ophthalmologist before any clinical decisions are made.

---

## 🌟 Why EyeCheck AI?

- **463 million** diabetics worldwide, 1 in 3 at risk of DR
- Rural India has **<1 ophthalmologist per 100,000 people**
- Specialist screening costs **₹2,000–5,000** per visit
- EyeCheck AI brings screening cost to **near zero**
- Works **offline** — deployable in any PHC or screening camp today

---

## 👩‍💻 Author

**Mahalakshmi N**
Biomedical Engineering, PSNA College of Engineering and Technology (2024–2028)

[

![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)

](https://www.linkedin.com/in/mahalakshmi-n-678522328)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>Early detection saves vision. AI makes it accessible for everyone.</i>
</p>
