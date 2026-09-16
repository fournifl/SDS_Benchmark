import geopandas as gpd
from pyogrio import list_layers
from pathlib import Path
from datetime import datetime

from concurrent.futures import ProcessPoolExecutor

def read_layer(layer):
    return layer, gpd.read_file(f_gpkg, layer=layer, engine="pyogrio", use_arrow=False, columns=["geometry", "date"])


# narrabeen
f_gpkg = Path('/home/florent/Projects/benchmark_satellite_coastlines/sentiline_results/NARRABEEN/Sentinel_2_coregistered/shoreline_results.gpkg')
# pour les shorelines landsat, il faudra concaténer les gdf des différentes missions Landsat 5 à 9
tmin = datetime(1976, 4, 27)
tmax = datetime(2019, 11, 28)

layers = list_layers(f_gpkg)[:, 0]
t_layers = [datetime.strptime(layer, '%Y%m%d_%Hh%M') for layer in layers]
layers = [layers[i] for i in range(len(t_layers)) if (t_layers[i] > tmin) & (t_layers[i] < tmax) ][0:5]


with ProcessPoolExecutor() as ex:
    results = dict(ex.map(read_layer, layers))

records = []
for layer, d in results.items():
    records.append({
        "date": d["date"].iloc[0],
        "geometry": d["geometry"].iloc[0],
    })

gdf = gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
gdf = gdf.to_crs(32756)
gdf.to_parquet(f_gpkg.parent / f_gpkg.name.replace('.gpkg', '.parquet'))
print(gdf)
# gdf = gpd.read_parquet("all_layers.parquet")
