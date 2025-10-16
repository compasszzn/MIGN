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

## 📦 Meta Data Download

The metadata are obtained from the **Global Summary of the Day (GSOD)** dataset provided by NOAA:  
🔗 [https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/](https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive/)

Please **manually download** the data for the years **2017–2024** and place them in the following directory: /realtime/Dataset/


## Data Processing Pipeline

Navigate to the processing directory:
```bash
cd data/process_data
# Filter Features; Note: Modify base_dir and output_dir in the script before running.
python data/process_data/step0_filter_feature.py
# Note: Modify folder_path in the script before running.
python data/process_data/step1_get_union_station.py
# Compute Climatology Statistics; Note: Modify base_path and output_path in the script before running.
python data/process_data/step2_climatology.py
```

### Data Preparation for MIGN
```
# One-step Input → One-step Output
python data/process_data/step3_generate_graph_dgl_step.py --input_day 1 --output_day 1
# Multi-step Input → Multi-step Output
python data/process_data/step3_generate_graph_dgl_multi_step.py --input_day 3 --output_day 4
```

### Data Preparation for Baselines
```
# One-step Input → One-step Output
python data/process_data/step4_generate_graph_pyg_step.py --input_day 1 --output_day 1
# Multi-step Input → Multi-step Outputt
python data/process_data/step4_generate_graph_pyg_multi_step.py --input_day 3 --output_day 4
```

### Run the code
```
python /home/zinanzheng/project/MIGN/main.py
```