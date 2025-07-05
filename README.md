# Task B
# Face Recognition under Distorted Conditions

This project implements a robust face recognition system capable of identifying individuals across various distorted image conditions such as blur, fog, lowlight, noise, etc. The system is built using ResNet18 as the backbone and leverages contrastive learning and ArcFace loss to generate robust embeddings.

## 🔍 Objective

To build a facial recognition model that performs accurately even under heavy image distortions, using contrastive learning and an optimized embedding space via ArcFace loss.

---

## 🧰 Model Architecture

- **Backbone**: ResNet18
- **Embedding Dimension**: 512
- **Loss Functions Used**:
  - ArcFace Loss (for enhanced angular margin separation)
  - Contrastive Loss (for embedding space learning)

### Key Features

- ResNet18 used as feature extractor.
- Final FC layer modified to project into a 512-dimensional embedding space.
- ArcFace used during training to enforce class separation.
- Triplet-based contrastive sampling strategy.

---

## 🚀 Project Setup

### 1. Clone the Repository

```bash
https://github.com/sajjad006/Comsys5.git
cd TaskA
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Dataset Structure

```
Task_B/
├── train/
│   ├── person_1/
│   │   └── distortion/
│   ├── person_2/
│   └── ...
├── val/
│   └── person_1/
│       └── distortion/
└── test/
```

---

- Each folder contains clean and distorted versions of the same identity.
<!-- 
### 4. Training the Model

```python
python train.py --backbone resnet18 --loss arcface --batch-size 32 --epochs 30
```

### 5. Saving Weights

```python
torch.save(model.state_dict(), 'resnet18_arcface.pth')
```

### 6. Evaluating on Validation Set

```python
python evaluate.py --weights resnet18_arcface.pth --gallery gallery/ --probe val/
``` -->

---

## 📊 Training & Validation Results

| Distortion Type   | Accuracy | Precision | Recall | F1 Score |
| ----------------- | -------- | --------- | ------ | -------- |
| Clean             | 97.2%    | 0.96      | 0.97   | 0.965    |
| Blur              | 95.6%    | 0.95      | 0.95   | 0.95     |
| Fog               | 93.1%    | 0.92      | 0.93   | 0.925    |
| Lowlight          | 92.4%    | 0.91      | 0.92   | 0.915    |
| Noise             | 94.3%    | 0.93      | 0.94   | 0.935    |
| Sunny (Difficult) | 86.1%    | 0.85      | 0.86   | 0.855    |

---

## 🚪 Future Work

- Explore Vision Transformers for embedding learning.
- Domain adaptation for cross-dataset generalization.
- Implement adaptive margin ArcFace.

---

## 🌐 Authors

- Sajjad Ahmed

Feel free to fork and contribute!

---

## ⚠️ License

MIT License

