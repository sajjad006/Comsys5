# Face Recognition with FaceNet and ArcFace Head

This project builds a face recognition system trained on clean and distorted facial images using a two-stage training pipeline with contrastive learning and ArcFace-based classification. The dataset contains 877 individuals with both clean and distorted images (blurred, foggy, lowlight, noisy, etc.).

---

## 📁 Dataset Structure

```
Task_B/
├── train/
│   ├── person_1/
│   │   ├── clean/
│   │   └── distortion/
│   ├── person_2/
│   └── ...
├── val/
│   └── person_1/
│       ├── clean/
│       └── distortion/
└── test/
```

---

## 🧠 Model Overview

- **Backbone**: Pretrained FaceNet (InceptionResNetV1 from `facenet-pytorch`)
- **Stage 1 Head**: ArcFace classification head (`ArcMarginProduct`)
- **Stage 2 Loss**: Contrastive Loss using face embeddings

---

## 🏋️ Stage 1: ArcFace Classification Fine-Tuning

### 🔹 Goal

Train ArcFace classification head with FaceNet backbone to recognize identity from clean and distorted training images.

### 🔹 Important Fixes

✅ **Shared label mapping** between train and val datasets:

```python
# Build train_ds and extract persons/label2idx
train_ds = FaceClassificationDataset('/content/Task_B/train', transform)
persons = train_ds.persons
label2idx = train_ds.label2idx

# Pass same mapping to val_ds
val_ds = FaceClassificationDatasetWithMapping('/content/Task_B/val', transform, persons, label2idx)
```

✅ **Optional**: Use clean-only images for validation classification accuracy:

```python
val_ds = FaceCleanValDataset('/content/Task_B/val', persons, label2idx, transform)
```

✅ **Better**: Use gallery-based evaluation for validation instead of classification accuracy:

```python
model.eval()
gallery = build_gallery(model, '/content/Task_B/val', transform, device)
acc, _ = evaluate(model, gallery, '/content/Task_B/val', transform, device)
```

---

## 🧪 Stage 2: Contrastive Learning Fine-Tuning

### 🔹 Goal

Fine-tune the FaceNet model to bring embeddings of distorted and clean images of the same person closer in embedding space.

### 🔹 Dataset Pairing

- For each person:
  - Anchor: clean image
  - Positive: distorted version (same ID)
  - Negative: image from a different person

### 🔹 Loss

```python
ContrastiveLoss(margin=1.0)
```

---

## ✅ Evaluation Strategy

Use a **gallery-based retrieval** approach:

1. Build a gallery of clean embeddings.
2. Extract embedding from each distorted image (query).
3. Compute cosine similarity with gallery.
4. Top-1 match should be of the same person.

---

## 🧃 Results and Tips

- Ensure **label mappings are consistent** across datasets.
- Don't rely on classification accuracy if your final goal is matching.
- **Visualize embeddings** using TSNE or cosine similarity matrix.
- Use a learning rate scheduler to improve convergence.

---

## 🧰 Requirements

- Python 3.8+
- PyTorch
- facenet-pytorch
- torchvision, PIL, numpy, tqdm

---

## 🚀 Run

```bash
# Stage 1: Classification Training
python train_stage1.py

# Stage 2: Contrastive Learning
python train_stage2.py

# Evaluate
python evaluate.py
```

---

## 👤 Author

- Sajjad Ahmed – Jadavpur University
