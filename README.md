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
```

## 🧠 Reward Function Design

To enable intelligent and fair client selection, FedRL uses a custom reward function for the reinforcement learning agent. The goal is to incentivize clients that contribute meaningfully to model improvement while promoting diversity and fairness.

The reward for each client is calculated as:

r = ΔAcc / [(1 + fc) * (1 + D_KL) * log(1 + |D|)]


**Where:**
- `ΔAcc` — The increase in global model accuracy after aggregating the client’s update.
- `fc` — The frequency of the client's participation in previous training rounds (to penalize over-selection).
- `D_KL` — The Kullback–Leibler divergence between the client’s local label distribution and the global distribution (to encourage statistical diversity).
- `|D|` — The size of the client's local dataset (scaled logarithmically to avoid bias toward large datasets).

This reward structure encourages the RL agent to:
- Select clients who contribute to better global accuracy
- Avoid overusing the same clients
- Promote data diversity from underrepresented distributions
- Ensure efficient and fair use of computational resources

## 📊 Results Summary

FedRL was benchmarked against the standard FedAvg strategy on both homogeneous and heterogeneous (non-IID) data splits using CIFAR-10 and MNIST datasets. The RL-based client selection approach consistently demonstrated faster convergence and higher final accuracy.

| Dataset           | Client Selection    | Final Accuracy | Convergence Speed      |
|------------------|---------------------|----------------|------------------------|
| CIFAR-10 (non-IID) | FedAvg (Random)     | 92.7%          | Baseline               |
|                  | FedRL (DQN-based)   | **93.26%**     | **37% faster**         |
| MNIST (non-IID)  | FedAvg (Random)     | 98.7%          | Baseline               |
|                  | FedRL (DQN-based)   | **99.5%**      | Lower compute per round |

FedRL achieved:
- Up to **12.5% improvement in test accuracy**
- **Significantly faster convergence** in non-IID scenarios
- **Reduced computational load** by selectively involving only high-value clients

## 🧰 Tech Stack

FedRL was built using a robust set of machine learning and systems tools to simulate federated environments and train reinforcement learning agents at scale.

- **Programming Languages**: Python 3.10
- **Deep Learning Frameworks**: PyTorch, TensorFlow (for auxiliary tools)
- **Federated Learning Framework**: [Flower](https://flower.dev/)
- **RL Algorithm**: Deep Q-Network (DQN)
- **Visualization**: TensorBoard, Matplotlib
- **Cluster & Cloud Tools**: Slurm (for HPC), AWS EC2, Docker
- **Libraries**: NumPy, Pandas, scikit-learn, OpenCV

The system is fully modular and easily extensible for:
- Supporting new datasets or data partitions
- Adding multi-agent or actor-critic RL extensions
- Incorporating additional system-level constraints (bandwidth, energy, etc.)
