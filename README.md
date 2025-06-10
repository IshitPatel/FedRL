# FedRL
A Reinforcement Learning Based Client Selection to optimize training computational overhead in Heterogenous Federated Learning

# 🧠 FedRL: A Reinforcement Learning Approach to Adaptive Client Selection in Federated Learning

*A research project by Ishit Patel | MS CS, San Francisco State University*

---

## 🔍 Overview

**FedRL** is a novel framework that integrates **Deep Q-Network (DQN)**-based reinforcement learning into **Federated Learning (FL)** to intelligently select clients for training. It addresses key challenges in FL such as:

- Non-IID data distribution  
- Unfair client participation  
- High communication and compute overhead

FedRL optimizes client selection in each round to maximize convergence speed and model performance, while ensuring fairness and diversity across participants.

---

## 🎯 Key Features

- ✅ DQN-based adaptive client selection
- 📉 37% faster convergence over FedAvg
- 📈 12.5% improvement in test accuracy
- ⚖️ Fairness-aware selection using participation frequency
- 🌍 KL-divergence-based heterogeneity measure
- ⚡ PyTorch + Flower-based scalable FL simulation

---

## 📚 Paper & Report

- 📄 [Read the project blog series on Medium](https://ishitpatel.medium.com)
- 📝 Full report (PDF available upon request)
- 📘 Title: *FedRL: A Reinforcement Learning Approach to Adaptive Client Selection in Federated Learning*

---

## 🧪 Datasets Used

- 🖼️ **CIFAR-10** with ResNet-18  
- ✍️ **MNIST** with SimpleCNN  
- Experiments on both **homogeneous** and **heterogeneous (non-IID)** data distributions

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/IshitPatel/FedRL.git
cd FedRL
