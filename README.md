# S-KANformer: Enhancing Transformers for Symbolic Calculations in High Energy Physics

This repository contains the source code for **SYMBA Test 2025**, developed as part of the project **[Next-Generation Transformer Models for Symbolic Calculations of Squared Amplitudes in High Energy Physics](https://ml4sci.org/gsoc/2025/proposal_SYMBA1.html).**

---
## 📂 Directory Structure

```
.
├── SineKAN/
│   ├── runs/
│   ├── config.py
│   ├── constants.py
│   ├── data.py
│   ├── fn_utils.py
│   ├── main.py
│   ├── model.py
│   ├── prefix_tokenizer.py
│   ├── seq_acc.ipynb
│   ├── seq_acc.py
│   ├── tokenizer.py
│   ├── trainer.py
├── Vanilla/
│   ├── runs/
│   ├── config.py
│   ├── constants.py
│   ├── data.py
│   ├── fn_utils.py
│   ├── main.py
│   ├── model.py
│   ├── prefix_tokenizer.py
│   ├── seq_acc.ipynb
│   ├── seq_acc.py
│   ├── tokenizer.py
│   ├── trainer.py
├── preprocess.ipynb
└── README.md
```

---
## 📌 Overview of Directories and Files

### **1️⃣ Modeling**
- **`SineKAN/`** – Implementation of the S-KANformer model, integrating SineKAN layers for enhanced symbolic representation.
- **`Vanilla/`** – Standard Transformer architecture used as a baseline for performance comparison.

### **2️⃣ Data Preprocessing**
- **`preprocess.ipynb`** – Prepares generated datasets before tokenization to ensure optimal input representation.

### **3️⃣ Model Training & Evaluation**
- **`seq_acc.ipynb`** – Runs sequence accuracy (Seq-Acc) calculation for evaluating model performance in single-GPU setups.
- **`seq_acc.py`** – Equivalent script for multi-GPU training setups.

### **4️⃣ Configuration & Constants (Present in Both Models)**
- **`config.py`** – Contains model training configurations, including hyperparameters and experiment settings.
- **`constants.py`** – Defines special tokens and tokenizer indices crucial for processing symbolic expressions.

### **5️⃣ Data Handling (Present in Both Models)**
- **`data.py`** – Handles dataset loading and processing for amplitude and squared amplitude expressions.

### **6️⃣ Utilities & Supporting Modules (Present in Both Models)**
- **`fn_utils.py`** – Houses helper functions to streamline model operations.
- **`tokenizer.py`** – Implements a custom tokenizer specialized for parsing amplitude expressions efficiently.

### **7️⃣ Model Implementation (Present in Both Models)**
- **`model.py`** – Defines the architectures for both the **S-KANformer** and **Vanilla Transformer** models.

### **8️⃣ Training & Inference (Present in Both Models)**
- **`trainer.py`** – Contains training and inference scripts tailored for efficient model deployment.

---
## 🛠 Training the Models

To get started with training, refer to the `runs/` directory inside `SineKAN/` and `Vanilla/`, which contain the necessary bash scripts for running experiments on both single-GPU and multi-GPU setups.

---



