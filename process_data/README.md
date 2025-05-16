# step1 

Go to the link https://www.ncei.noaa.gov/data/global-summary-of-the-day/archive download data the data from 2017-2024.

# step2

Filter the file into daily format

python step0_filter_feature.py

# step3

get all stations in a csv file

python step1_get_union_station.py

# step4

compute the mean and std of data

python step2_climatology.py

# step5

generate data for dgl(for our model) and pyg(for baselines) respectively.

python step3_generate_graph_dgl_step.py
python step4_generate_graph_pyg_step.py