
import matplotlib as plt

# conda install -c https://conda.anaconda.org/ioos geopandas
# conda install -c conda-forge shapely
# conda install -c conda-forge descartes

import geopandas as gpd
from shapely.geometry import Point, Polygon
import descartes

#final_crs = {'init': 'epsg:28992'}

buildings = gpd.read_file(
    'Ithaca/map/buildings-polygon.shp')  # .to_crs(final_crs)
roads = gpd.read_file('Ithaca/map/roads-line.shp')  # .to_crs(final_crs)
landcover = gpd.read_file(
    'Ithaca/map/landcover-polygon.shp')  # .to_crs(final_crs)
# .to_crs(final_crs)
water = gpd.read_file('Ithaca/map/water_areas-polygon.shp')
waterlines = gpd.read_file(
    'Ithaca/map/water_lines-line.shp')  # .to_crs(final_crs)

fig, ax = plt.subplots(figsize=(10, 10))

roads.plot(ax=ax, color="grey")
landcover.plot(ax=ax, color="green", alpha=0.03)
water.plot(ax=ax, color="blue", alpha=0.1)
waterlines.plot(ax=ax, color="blue", alpha=0.1)
buildings.plot(ax=ax, color="black")

minx, miny, maxx, maxy = -76.4900, 42.443484, -76.475089, 42.4555
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)

DL1 = Point((-76.485089, 42.450960))
DL2 = Point((-76.481194, 42.450536))
DL3 = Point((-76.484268, 42.448039))
DL4 = Point((-76.483743, 42.451245))

gdf_stations = gpd.GeoSeries([DL1, DL2, DL3, DL4])
gdf_stations.plot(ax=ax, markersize=50, color="red", marker="o")

# Annotate with names

stations = [
    "DL1",
    "DL2",
    "DL3",
    "DL4"
]
xx = 0.0003
yy = 0.0001

for i, geo in gdf_stations.centroid.iteritems():
    ax.annotate(s=stations[i], xy=[geo.x+xx, geo.y-yy], color="red")


# Save PDF
plt.savefig('Map.pdf')