# 🌍 Mesh Interpolation Graph Network for Dynamic and Spatially Irregular Global Weather Forecasting

This is the **official PyTorch Lightning implementation** of the paper:  
**Mesh Interpolation Graph Network for Dynamic and Spatially Irregular Global Weather Forecasting**, accepted at **NeurIPS 2025**.  
[[📄 arXiv]](https://arxiv.org/pdf/2509.20911)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/hanjq17/GMN/blob/main/LICENSE)

---

## ⚙️ Requirements

### 🧩 Core Implementation (DGL-based)

MIGN is implemented using **[DGL (Deep Graph Library)](https://www.dgl.ai/)**.  

```bash
# Create and activate a clean conda environment
conda create -n dgl python=3.10
conda activate dgl

# Install PyTorch (CUDA 12.1)
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Install DGL (CUDA 12.1 compatible)
pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html

# Install additional dependencies
pip install numpy==1.26.0
pip install healpy
pip install lightning
pip install -U 'wandb>=0.12.10'
pip install jupyter
```

### 📊 Baselines (PyTorch Geometric Temporal & Torch Spatiotemporal)

We also provide baseline implementations built upon **two widely used spatiotemporal graph learning frameworks**:

- **[PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal)**  
- **[Torch Spatiotemporal (TSL)](https://github.com/TorchSpatiotemporal/tsl)**  

To install the dependencies required for running these baselines:

```bash
# Install PyTorch Geometric and related packages (compatible with CUDA 12.1)
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
# Install PyTorch Geometric Temporal (local version)
cd baseline/spatial_temporal/pytorch_geometric_temporal/
pip install -e .
# Install Torch Spatiotemporal (TSL)
pip install torch-spatiotemporal
```