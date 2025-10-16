import torch
import argparse
import pandas as pd
from pathlib import Path
from locationencoder.pe import SphericalHarmonics
import time

level=3
# Initialize the Spherical Harmonics model
sh = SphericalHarmonics(legendre_polys=level)  # Adjust degree of Legendre polynomials if needed

# Variables to iterate over, including the ones you're interested in
variables = [
    'accumulated_precipitation',
    # # 'air_temperature_mean',
    # 'air_temperature_min',
    # 'air_temperature_max',
    # # 'snow_depth',
    # 'fresh_snow',
    # 'wind_speed'
]

for variable in variables:
    # Read the CSV file for the current variable
    station_data = pd.read_csv(f"/data/zzn/insitu/insitu_daily_st/unique_primary_station_ids_{variable}.csv")
    
    # Extract latitude and longitude for each station
    latitudes = station_data['latitude'].values
    longitudes = station_data['longitude'].values
    
    # Create a list to hold embeddings for this variable
    embeddings_for_variable = []
    t1 = time.time()
    # Process each station's coordinates and compute embedding
    for lat, lon in zip(latitudes, longitudes):
        lonlat = torch.tensor([[lon, lat]], dtype=torch.float32)  # Longitude first, then Latitude
        embedded_lonlat = sh(lonlat)  # Get the embedding from Spherical Harmonics
        # Append the resulting embedding to the list
        embeddings_for_variable.append(embedded_lonlat.squeeze(0))  # Remove the extra dimension
    # Convert the list of embeddings into a 2D tensor for this variable
    embeddings_tensor = torch.stack(embeddings_for_variable)  # Shape: [num_stations, embedding_dim]
    t2 = time.time()
    print(t2-t1)
    # Save the embeddings for this variable to a file
    # embeddings_file = Path('/data/zzn/insitu/insitu_daily_st') / f'{variable}_embeddings_level_{level}.pt'
    # torch.save(embeddings_tensor, embeddings_file)
    # print(f"Embeddings for {variable} saved to {embeddings_file}")

