import healpy as hp
import numpy as np
import pandas as pd

# Define the nside parameter for Healpix
nside = 2 ** 5
n_pixels = hp.nside2npix(nside)

# Get theta and phi (the angular coordinates for the Healpix grid)
theta, phi = hp.pix2ang(nside, np.arange(n_pixels))

# Convert theta and phi to latitude and longitude
latitudes = 90 - np.degrees(theta)  # Convert from theta to latitude
longitudes = np.degrees(phi) - 180  # Convert from phi to longitude, ensuring the range is [-180, 180]

# Create a DataFrame
df = pd.DataFrame({
    'primary_station_id': np.arange(n_pixels),  # Use pixel index as station ID
    'latitude': latitudes,
    'longitude': longitudes
})

# Save the DataFrame to a CSV file
output_file = 'unique_primary_station_ids_healpix.csv'
base_path = '/data/zzn/insitu/insitu_daily_filter_7_14'
df.to_csv(f"{base_path}/{output_file}", index=False)

print(f"CSV file saved as {output_file}")