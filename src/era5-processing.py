"""
ERA5 Preprocessing.
"""

import xarray as xr

ds = xr.load_dataset('data/raw/43b6f072cb65230a2e1c2354af9a06c7.grib', engine='cfgrib')

# u_v_wind = ds[['u', 'v']]
df = ds.to_dataframe()

# df.to_csv('era5_windspeeds.csv', index=False)
