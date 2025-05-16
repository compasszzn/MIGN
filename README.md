```
conda create -n dgl python=3.10
conda activate dgl
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html
pip install numpy==1.26.0
pip install cfgrib
pip install lightning
pip install -U 'wandb>=0.12.10'
pip install jupyter

pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
cd baseline/spatial_temporal
```

```
cd MIGN
without sh embedding
python main.py 

see process_data for getting data

with sh embedding
python main.py --sh_after --sh_before
```