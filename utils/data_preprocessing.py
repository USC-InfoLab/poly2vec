import pandas as pd
import geopandas as gpd
import torch
import numpy as np
import os
import random
random.seed(2024)
import sys

sys.path.append("../")
from models.fourier_encoder import GeometryFourierEncoder


def normalize_coordinate(value, min_val, max_val):
    """Normalize a coordinate to the range [0, 2]."""
    """Add -1 to shift the range to [-1, 1]."""
    return 2 * (value - min_val) / (max_val - min_val) - 1


def normalize_geometry_polygon(geometry, min_x, max_x, min_y, max_y):
    if geometry.is_empty:
        return geometry
    normalized_coords = [(normalize_coordinate(x, min_x, max_x), normalize_coordinate(y, min_y, max_y)) for x, y in
                         geometry.exterior.coords]
    return normalized_coords


def normalize_geometry_linestring(geometry, min_x, max_x, min_y, max_y):
    if geometry.is_empty:
        return geometry
    normalized_coords = [(normalize_coordinate(x, min_x, max_x), normalize_coordinate(y, min_y, max_y)) for x, y in
                         geometry.coords]
    return normalized_coords


def read_urban_dataset(root_dir, dataset_name):
    # Load original dataset file
    # TODO: We need to unify the file_name, the type of objects saved in each file,
    #  and the column name of the geometry in the original data file and the gdf objects.
    # region (some of the regions could be 3D)
    if dataset_name == "NewYork":
        regions_file = f"{root_dir}/{dataset_name}/regions.pkl"
        regions_pd = pd.DataFrame(pd.read_pickle(regions_file))
        regions_pd = regions_pd.rename(columns={"shape": "geometry"})
        # TODO: Even if we claim geometry is this region_geometry_marker (e.g., "shape" for NewYork,
        #  the created gdf object still doesn't have "geometry" column but only "shape" column)
        regions_gdf = gpd.GeoDataFrame(regions_pd, geometry=r"geometry")
        regions_gdf["region_id"] = regions_gdf.index
    elif dataset_name == "Singapore":
        regions_file = f"{root_dir}/{dataset_name}/regions_updated.pkl"
        regions_gdf = pd.read_pickle(regions_file)
        regions_gdf["region_id"] = regions_gdf.index
    # buildings
    if dataset_name == "NewYork":
        buildings_file = f"{root_dir}/{dataset_name}/buildings.pkl"
        buildings_gdf = pd.read_pickle(buildings_file)
        buildings_gdf["building_id"] = buildings_gdf.index
    elif dataset_name == "Singapore":
        buildings_file = f"{root_dir}/{dataset_name}/building.pkl"
        buildings_pd = pd.DataFrame(pd.read_pickle(buildings_file))
        buildings_pd = buildings_pd.rename(columns={"shape": "geometry"})
        buildings_gdf = gpd.GeoDataFrame(buildings_pd, geometry="geometry")
        buildings_gdf["building_id"] = buildings_gdf.index
    # poi
    poi_file = f"{root_dir}/{dataset_name}/poi_updated.pkl"
    poi_gdf = pd.read_pickle(poi_file)
    poi_gdf["poi_id"] = poi_gdf.index
    # roads
    roads_file = f"{root_dir}/{dataset_name}/roads.pkl"
    if dataset_name == "NewYork":
        roads_gdf = pd.read_pickle(roads_file)
    elif dataset_name == "Singapore":
        roads_pd = pd.read_pickle(roads_file)
        #roads_pd = roads_pd.rename(columns={"shape": "geometry"})
        #TODO: For Singapore, its road.pkl contains two types of geometries, "geometry" is the GPS coordinates system, while "geometry" is its own coordinates system
        roads_gdf = gpd.GeoDataFrame(roads_pd, geometry="shape")
    roads_gdf["road_id"] = roads_gdf.index

    return regions_gdf, buildings_gdf, poi_gdf, roads_gdf


def urban_dataset_process(root_dir, dataset_name):
    print(f"Processing {dataset_name}...")
    # Load original dataset file
    regions_gdf, buildings_gdf, poi_gdf, roads_gdf = read_urban_dataset(root_dir, dataset_name)

    # Get the regions' bounds
    min_x, min_y, max_x, max_y = regions_gdf["geometry"].total_bounds

    # POIs (points)
    print(f"POI (points)")
    # Normalize the coordinates
    poi_gdf["coordinates"] = poi_gdf["geometry"].apply(lambda geom: list(geom.coords[0]))
    poi_gdf['normalized_coordinates'] = poi_gdf['coordinates'].apply(
        lambda coord: [normalize_coordinate(coord[0], min_x, max_x), normalize_coordinate(coord[1], min_y, max_y)])
    poi_gdf.to_pickle(f"{root_dir}/{dataset_name}/poi_normalized.pkl")
    print("POI normalization finished.")

    pois_to_regions = {}
    # Loop through each region and find the pois that fall within it
    for idx, region in regions_gdf.iterrows():
        # Get the region's geometry
        region_geom = region["geometry"]
        # Use spatial join to find pois within this region
        pois_in_region = poi_gdf[poi_gdf.geometry.within(region_geom)]
        # Assign the poi IDs or geometries to the corresponding region in the dictionary
        pois_to_regions[region['region_id']] = pois_in_region['poi_id'].tolist()
    # add the list of pois to the regions_gdf
    regions_gdf['pois'] = regions_gdf['region_id'].map(pois_to_regions)
    print("Incorporated POI into regions")

    # Roads (polylines)
    print("Roads (polylines)")
    roads_gdf['coordinates'] = roads_gdf['geometry'].apply(lambda geom: list(geom.coords))
    roads_gdf['normalized_coordinates'] = roads_gdf['geometry'].apply(
        lambda geom: normalize_geometry_linestring(geom, min_x, max_x, min_y, max_y))
    print("Roads normalization finished.")
    # pad normalized coordinates to max length
    max_length = max(roads_gdf['normalized_coordinates'].apply(len))
    roads_gdf['padded_coordinates'] = roads_gdf['coordinates'].apply(
        lambda coords: coords + [(0, 0)] * (max_length - len(coords)))
    roads_gdf['padded_norm_coordinates'] = roads_gdf['normalized_coordinates'].apply(
        lambda coords: coords + [(0, 0)] * (max_length - len(coords)))
    roads_gdf['len'] = roads_gdf['normalized_coordinates'].apply(len)
    print(
        f"After padding to max length {max_length}, we add {max_length * roads_gdf.shape[0] - roads_gdf['len'].sum()} points to {roads_gdf.shape[0]} roads. "
        f"Invalid ratio is {(max_length * roads_gdf.shape[0] - roads_gdf['len'].sum()) / roads_gdf.shape[0]}")
    # save roads
    roads_gdf.to_pickle(f"{root_dir}/{dataset_name}/roads_normalized.pkl")

    roads_to_regions = {}
    # Loop through each region and find the roads that fall within it
    for idx, region in regions_gdf.iterrows():
        # Get the region's geometry
        region_geom = region["geometry"]
        # Use spatial join to find roads within this region
        roads_in_region = roads_gdf[roads_gdf.geometry.within(region_geom)]
        # Assign the road IDs or geometries to the corresponding region in the dictionary
        roads_to_regions[region['region_id']] = roads_in_region['road_id'].tolist()
    # add the list of roads to the regions_gdf
    regions_gdf['roads'] = regions_gdf['region_id'].map(roads_to_regions)
    print("Incorporated roads into regions")

    # Buildings (polygons)
    print("Buildings (polygons)")
    # simplify geometries of buildings
    buildings_gdf['simple_shape'] = buildings_gdf['geometry'].simplify(tolerance=0.1)
    buildings_gdf['coordinates'] = buildings_gdf['simple_shape'].apply(lambda geom: list(geom.exterior.coords))
    buildings_gdf['normalized_coordinates'] = buildings_gdf['simple_shape'].apply(
        lambda geom: normalize_geometry_polygon(geom, min_x, max_x, min_y, max_y))
    print("Buildings normalization finished.")
    # pad normalized coordinates to max length
    max_length = max(buildings_gdf['normalized_coordinates'].apply(len))
    buildings_gdf['padded_coordinates'] = buildings_gdf['coordinates'].apply(
        lambda coords: coords + [(0, 0)] * (max_length - len(coords)))
    buildings_gdf['padded_norm_coordinates'] = buildings_gdf['normalized_coordinates'].apply(
        lambda coords: coords + [(0, 0)] * (max_length - len(coords)))
    buildings_gdf['len'] = buildings_gdf['normalized_coordinates'].apply(len)
    # TODO: There are some invalid polygons in NewYork dataset
    buildings_gdf = buildings_gdf.loc[buildings_gdf.geometry.is_valid]
    buildings_gdf = buildings_gdf.reset_index(drop=True)
    # Don't forget to reset ["building_id"] after reset index.
    buildings_gdf["building_id"] = buildings_gdf.index
    print(
        f"After padding to max length {max_length}, we add {max_length * buildings_gdf.shape[0] - buildings_gdf['len'].sum()} points to {buildings_gdf.shape[0]} buildings. "
        f"Invalid ratio is {(max_length * buildings_gdf.shape[0] - buildings_gdf['len'].sum()) / buildings_gdf.shape[0]}")
    # save the updated buildings data with normalized coordinates
    buildings_gdf.to_pickle(f"{root_dir}/{dataset_name}/buildings_normalized.pkl")

    buildings_to_regions = {}
    # Loop through each region and find the buildings that fall within it
    for idx, region in regions_gdf.iterrows():
        # Get the region's geometry
        region_geom = region["geometry"]
        # Use spatial join to find buildings within this region
        buildings_in_region = buildings_gdf[buildings_gdf.geometry.within(region_geom)]
        # Assign the building IDs or geometries to the corresponding region in the dictionary
        buildings_to_regions[region['region_id']] = buildings_in_region['building_id'].tolist()
    regions_gdf['buildings'] = regions_gdf['region_id'].map(buildings_to_regions)
    print("Incorporated buildings into regions")

    # save regions_gdf with all the associated data
    regions_gdf.to_pickle(f"{root_dir}/{dataset_name}/regions_with_pois_buildings_roads.pkl")


if __name__ == "__main__":
    root_dir = "./Poly2Vec/data"
    for dataset_name in ["NewYork", "Singapore"]:
        print(dataset_name)
        urban_dataset_process(root_dir, dataset_name)


