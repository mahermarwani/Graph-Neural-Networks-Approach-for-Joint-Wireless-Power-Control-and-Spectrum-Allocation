---

# 📡 Graph Neural Networks for Joint Wireless Power Control and Spectrum Allocation

Welcome to the official repository for:

> **Graph Neural Networks Approach for Joint Wireless Power Control and Spectrum Allocation**  
> *By Maher Marwani and Georges Kaddoum*  
> Published in *IEEE Transactions on Machine Learning in Communications and Networking*, 2024

---

## 🔬 Overview

This work introduces a novel **Graph Neural Network (GNN)-based framework** for tackling the **joint power control and spectrum allocation** problem in wireless communication networks. It targets Device-to-Device (D2D) communication in complex interference environments where traditional resource allocation techniques fall short due to the non-Euclidean structure of wireless topologies.

---

## 📊 Results & Visualizations

### 📍 Simulation Setup:  
- **Area**: 500m x 500m  
- **Topology**: 50 D2D links  
- **Minimum data rate constraint**: `1000 Kbps`  
- **Bandwidth**: 10 Resource Blocks (RBs) of 500 Hz each  

### 📈 Performance Visualization  
The following animation showcases the **evolution of data rates** during the simulation:

![Rate Animation](rate_animation.gif)

This visual captures the **adaptive behavior** of the GNN-based solution, effectively managing **interference**, **power**, and **spectrum resources** over time.

---

## 📄 Citation

If you find this work useful in your research or applications, please cite it using the following BibTeX entry:

```bibtex
@ARTICLE{10545547,
  author={Marwani, Maher and Kaddoum, Georges},
  journal={IEEE Transactions on Machine Learning in Communications and Networking}, 
  title={Graph Neural Networks Approach for Joint Wireless Power Control and Spectrum Allocation}, 
  year={2024},
  volume={2},
  pages={717-732},
  doi={10.1109/TMLCN.2024.3408723}
}
```

---

## 🧠 Abstract

The rising complexity of wireless environments driven by modern applications and user demands challenges traditional Radio Resource Management (RRM) frameworks. Although Deep Learning (DL) approaches offer adaptive solutions, most are limited to **Euclidean data structures**, ignoring the **graph-based nature** of wireless topologies.

This work proposes a GNN-based model that directly operates on **non-Euclidean representations** of wireless networks, enabling efficient joint optimization of power control and spectrum allocation. The framework:
- Adapts to varying interference conditions
- Supports flexible bandwidth allocation
- Maintains robust performance in imperfect channel conditions  
Experimental results demonstrate clear advantages in **convergence speed**, **generalization**, and **robustness** over existing solutions.

---

## ⚙️ Installation & Usage

Follow the steps below to install dependencies and run simulations:

### 🔧 Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mahermarwani/Graph-Neural-Networks-Approach-for-Joint-Wireless-Power-Control-and-Spectrum-Allocation.git
   cd Graph-Neural-Networks-Approach-for-Joint-Wireless-Power-Control-and-Spectrum-Allocation
   ```

2. **Install Required Packages**
   ```bash
   pip install -r requirements.txt
   ```

---

### 🚀 Running the Code

1. **Generate Wireless Network Topology**
   ```bash
   python wireless/Network.py
   ```

2. **Generate Training & Testing Data**
   ```bash
   python data_generation.py
   ```

3. **Train the GNN Model**
   ```bash
   python training.py
   ```

4. **Evaluate the Model**
   ```bash
   python testing.py
   ```

5. **Optional: Solve Using Genetic Algorithm**
   ```bash
   python GA_solver.py
   ```

---

## 📬 Questions or Feedback?

Feel free to open an [Issue](https://github.com/mahermarwani/Graph-Neural-Networks-Approach-for-Joint-Wireless-Power-Control-and-Spectrum-Allocation/issues) or submit a [Pull Request](https://github.com/mahermarwani/Graph-Neural-Networks-Approach-for-Joint-Wireless-Power-Control-and-Spectrum-Allocation/pulls) — contributions are welcome!

---
