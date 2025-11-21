# Cross-Dataset CNN Collaboration Report

Team: Team-08 (Anmol and Balveer Singh) 

Users:  
- **User 1 (Model v1)** – Dataset A
- **User 2 (Model v2)** – Dataset B

---

## 1. Task Description

The goal of this project is to collaboratively design, train, and evaluate convolutional neural network (CNN) models for the **same image classification task (cats vs dogs)** using **different datasets that cannot be shared**.

Each teammate:

- Trains a CNN model on their own dataset.
- Shares only the trained model weights and metrics.
- Evaluates the other teammate’s model on their own dataset.
- Compares in-domain vs cross-domain performance to study **generalization** and **domain shift**.

---

## 2. Datasets

### 2.1 User 1 Dataset (Model v1)

- **Name:** Dogs vs Cats Redux
- **Source:** Kaggle
- **Task:** Binary classification – cat vs dog
- **Train size:** 4000 cats + 4005 dogs
- **Test/Val size:** 1010 cats + 1012 dogs

### 2.2 User 2 Dataset (Model v2)

- **Name:** Dogs vs Cats Redux
- **Source:** Kaggle  
- **Task:** Binary classification – cat vs dog  
- **Train size:** 12500 cats + 12500 dogs  
- **Val/Test size:** 12500


---

## 3. Model Architectures

### 3.1 Model v1 (User 1)

- **Type:** Simple CNN
- **Key details:**
    - Conv(3→32) + BN + ReLU + MaxPool
    - Conv(32→64) + BN + ReLU + MaxPool
    - Conv(64→128) + BN + ReLU + AdaptiveAvgPool
    - FC: 128 → 64 → num_classes
    - Input size: 224×224
    - Optimizer & LR schedule: Adam, 1e-3

### 3.2 Model v2 (User 2 – Custom CNN)

- **Type:** Custom CNN implemented in `models/model_v2.py`
- **Architecture summary:**
  - 3 convolutional blocks:
    - Conv → BatchNorm → ReLU → MaxPool (×3)
  - Global average pooling using `AdaptiveAvgPool2d(1,1)`
  - Fully connected classifier:
    - Linear → ReLU → Dropout → Linear (num_classes)
  - Input size: `IMAGE_SIZE` (e.g., 224×224 or 256×256)
- **Training setup:**
  - Loss: CrossEntropyLoss  
  - Optimizer: Adam (e.g., `lr = 1e-3`, with optional weight decay)  
  - Scheduler: StepLR
  - Batch size: 32  
  - Epochs: 35  

---

## 4. Training & Evaluation Setup

- Framework: **PyTorch**
- Data loading:
  - User 1: ImageFolder / custom loader (TODO)
  - User 2: ImageFolder for training, CSV-based dataset for test.
- Common transforms (approx):
  - Resize to `256 × 256`
  - Random horizontal flip, rotation, color jitter (for training)
  - Normalization with ImageNet mean/std
- Metrics:
  - Accuracy
  - Macro F1-score
  - Confusion matrix (for per-class behavior)

---

## 5. Results

### 5.1 In-Domain Performance

| Model      | Trained On (User) | Evaluated On | Accuracy | F1 (macro) | Notes                |
|-----------|--------------------|--------------|----------|-----------|----------------------|
| Model v1  | User 1 dataset     | User 1 data  | 73.79     | 73.74      | In-domain baseline   |
| Model v2  | User 2 dataset     | User 2 data  | 84.44     | 84.43      | In-domain baseline   |

### 5.2 Cross-Dataset Performance

| Model      | Trained On | Evaluated On (Other User) | Accuracy | F1 (macro) | Notes                    |
|-----------|------------|----------------------------|----------|-----------|--------------------------|
| Model v1  | User 1     | User 2 data                | 76.5     | 76.42      | Tests generalization     |
| Model v2  | User 2     | User 1 data                | 85.52       | 85.49      | Tests generalization     |


---
## 6. Observations on Generalization & Domain Shift

- **Model v1** performs lower both in-domain and cross-domain mainly due to:
  - A **smaller training dataset**
  - A **simpler CNN architecture** with limited feature capacity
  - Insufficient training epochs

- **Model v2** shows better generalization because:
  - It was trained on a **larger and more diverse dataset**
  - Uses a **stronger architecture** with dropout and scheduling
  - Has more stable feature extraction, leading to higher cross-domain accuracy

- The performance gap between the two models is mostly explained by differences in:
  - Training data size  
  - Model complexity  
  - Training duration

---

## 7. Conclusion & Future Work

- Model v2 generalizes better due to **more training data** and a **stronger architecture**.
- Model v1 underperforms because it was trained with **fewer samples** and a **simpler, less expressive CNN**.
- Future improvements:
  - Increase training data or augmentations
  - Use deeper architectures or pretrained models
  - Train for more epochs with proper regularization

*This report summarizes the collaborative CNN experiment, datasets, models, and cross-dataset evaluation carried out by User 1 and User 2 for the cat vs dog classification task.*
