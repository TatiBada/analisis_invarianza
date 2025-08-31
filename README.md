# Analysis of Invariance

This repository contains the code and experiments for the Master’s thesis **"Analysis of Invariance in Modern Computer Vision Models"**.  
The project investigates how different neural network architectures—**EfficientNet-B0** and **FasterViT**—handle a diverse set of **geometric and color transformations** under various training strategies.

---

## 📌 Overview
- **Architectures**: EfficientNet-B0 (CNN-based) and FasterViT (Transformer-based).
- **Training strategies**:  
  - Full training (from scratch)  
  - Fine-tuning  
  - Transfer learning  
- **Dataset**: Subset of TinyImageNet (200k training images, 1k evaluation images).  
- **Transformations**: 10 types (4 geometric + 6 color) with 8 parameter variations each.  
- **Metrics**: Invariance measured at both block-level and output-level.

---

## 📂 Repository Structure

