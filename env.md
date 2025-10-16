conda create -n insitu python==3.9
conda activate insitu
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
cd pytorch_geometric_temporal/
pip install -e .
pip install jupyter