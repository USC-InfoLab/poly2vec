import pandas as pd
import torch
import math
from shapely import Polygon, LineString, Point
from shapely.affinity import translate, scale
from shapely.ops import nearest_points
import pickle
import numpy as np
import random

random.seed(2024)
import sys

sys.path.append("../")
from copy import deepcopy as c

precision = 10
np.set_printoptions(precision=precision)


def geometry_format_convert(geometry, geometry_type):
    if geometry_type == "polygon":
        ori_coords = list(geometry.exterior.coords)
        result_coords = [[round(x, precision), round(y, precision)] for x, y in ori_coords]
    elif geometry_type == "polyline":
        ori_coords = list(geometry.coords)
        result_coords = [[round(x, precision), round(y, precision)] for x, y in ori_coords]
    elif geometry_type == "point":
        ori_coords = list(geometry.coords)
        result_coords = [round(ori_coords[0][0], precision), round(ori_coords[0][1], precision)]

    return result_coords


def load_dataset(dataset_name, geometry_type):
    if geometry_type == "polygon":
        geometry_gdf = pd.read_pickle(f"{root_dir}/{dataset_name}/buildings_normalized.pkl")
        geometries = geometry_gdf["padded_norm_coordinates"].tolist()
        lengths = geometry_gdf["len"].tolist()
        num_geometry = len(geometry_gdf)
    elif geometry_type == "polyline":
        geometry_gdf = pd.read_pickle(f"{root_dir}/{dataset_name}/roads_normalized.pkl")
        geometries = geometry_gdf["padded_norm_coordinates"].tolist()
        lengths = geometry_gdf["len"].tolist()
        num_geometry = len(geometry_gdf)
    elif geometry_type == "point":
        geometry_gdf = pd.read_pickle(f"{root_dir}/{dataset_name}/poi_normalized.pkl")
        geometries = geometry_gdf["normalized_coordinates"].tolist()
        lengths = [2 for i in range(len(geometries))]
        num_geometry = len(geometry_gdf)

    return geometries, lengths, num_geometry


def list2shapely(list_coords, length, geometry_type):
    if geometry_type == "polygon":
        return Polygon(list_coords[:length])
    elif geometry_type == "polyline":
        return LineString(list_coords[:length])
    elif geometry_type == "point":
        return Point(list_coords)
    else:
        print(f"Unsupported geometry_type: {geometry_type}.")

def check_relationship(geom1, geom2, geometry_type1, geometry_type2, relationship):
    if geometry_type1 == "polygon" and geometry_type2 == "polygon" and relationship == "intersect":
        return geom1.intersects(geom2)
    elif geometry_type1 == "polygon" and geometry_type2 == "polyline" and relationship == "intersect":
        return geom1.intersects(geom2)
    elif geometry_type1 == "polygon" and geometry_type2 == "point" and relationship == "contain":
        return geom1.contains(geom2)
    elif geometry_type1 == "polyline" and geometry_type2 == "polyline" and relationship == "intersect":
        return geom1.intersects(geom2)
    elif geometry_type1 == "polyline" and geometry_type2 == "point" and relationship == "contain":
        return geom1.contains(geom2)
    else:
        print(f"Unsupported {relationship} check between {geometry_type1} and {geometry_type2}")


def pairwise_relationship(dataset_name, geometry_type1, geometry_type2, relationship="intersect", num_pairs=5000):
    geometries1, lengths1, num_geometries1 = load_dataset(dataset_name, geometry_type1)
    geometries2, lengths2, num_geometries2 = load_dataset(dataset_name, geometry_type2)
    print(
        f"Find {num_pairs} of {relationship} pairs from {num_geometries1} {geometry_type1} and {num_geometries2} {geometry_type2}.")

    # Randomly find num_pair non-intersect pairs.
    neg_pairs = []
    neg_geom1_coords = []
    neg_geom2_coords = []
    neg_lengths1 = []
    neg_lengths2 = []
    while len(neg_pairs) < num_pairs:
        idx1 = random.randint(0, min(num_pairs, num_geometries1) - 1)
        idx2 = random.randint(0, min(num_pairs, num_geometries2) - 1)
        geom1 = list2shapely(geometries1[idx1], lengths1[idx1], geometry_type1)
        geom2 = list2shapely(geometries2[idx2], lengths2[idx2], geometry_type2)
        if not check_relationship(geom1, geom2, geometry_type1, geometry_type2, relationship) and [idx1,
                                                                                                   idx2] not in neg_pairs:
            neg_pairs.append([idx1, idx2])
            neg_geom1_coords.append(torch.tensor(geometries1[idx1], dtype=torch.float32))
            neg_geom2_coords.append(torch.tensor(geometries2[idx2], dtype=torch.float32))
            neg_lengths1.append(lengths1[idx1])
            neg_lengths2.append(lengths2[idx2])

    # Pos pairs are much rarer than neg pairs. If generating randomly it could take days.
    # Instead, we randomly pick two polygons and shift them to intersect.
    pos_pairs = []
    pos_geom1_coords = []
    pos_geom2_coords = []
    pos_lengths1 = []
    pos_lengths2 = []
    while len(pos_pairs) < num_pairs:
        idx1 = random.randint(0, min(num_pairs, num_geometries1) - 1)
        idx2 = random.randint(0, min(num_pairs, num_geometries2) - 1)
        geom1 = list2shapely(geometries1[idx1], lengths1[idx1], geometry_type1)
        geom2 = list2shapely(geometries2[idx2], lengths2[idx2], geometry_type2)
        if check_relationship(geom1, geom2, geometry_type1, geometry_type2, relationship) and [idx1,
                                                                                               idx2] not in pos_pairs:
            pos_pairs.append([idx1, idx2])
            #print(f"Added {len(pos_pairs)} pos pairs.")
            pos_geom1_coords.append(torch.tensor(geometries1[idx1], dtype=torch.float32))
            pos_geom2_coords.append(torch.tensor(geometries2[idx2], dtype=torch.float32))
            pos_lengths1.append(lengths1[idx1])
            pos_lengths2.append(lengths2[idx2])
        if not check_relationship(geom1, geom2, geometry_type1, geometry_type2, relationship) and [idx1,
                                                                                                   idx2] not in pos_pairs:
            point1, point2 = nearest_points(geom1, geom2)
            shift_x = point1.x - point2.x
            shift_y = point1.y - point2.y
            geom2_shifted = translate(geom2, xoff=shift_x, yoff=shift_y)
            if check_relationship(geom1, geom2_shifted, geometry_type1, geometry_type2, relationship):
                pos_pairs.append([idx1, idx2])
                #print(f"Added {len(pos_pairs)} pos pairs.")
                geom2_coords = c(geometries2[idx2])
                geom2_coords[:lengths2[idx2]] = geometry_format_convert(geom2_shifted, geometry_type2)
                pos_geom1_coords.append(torch.tensor(geometries1[idx1], dtype=torch.float32))
                pos_geom2_coords.append(torch.tensor(geom2_coords, dtype=torch.float32))
                pos_lengths1.append(lengths1[idx1])
                pos_lengths2.append(lengths2[idx2])
            else:
                # If they still don't intersect, adjust slightly in the shift direction
                adjustment_x, adjustment_y = shift_x * 0.1, shift_y * 0.1  # Small adjustment factor
                max_attempts = 10  # Limit further attempts
                attempt = 0
                while not check_relationship(geom1, geom2_shifted, geometry_type1, geometry_type2,
                                             relationship) and attempt < max_attempts:
                    geom2_shifted = translate(geom2_shifted, xoff=adjustment_x, yoff=adjustment_y)
                    attempt += 1
                    # Check if intersected after adjustment
                    if check_relationship(geom1, geom2_shifted, geometry_type1, geometry_type2, relationship):
                        pos_pairs.append([idx1, idx2])
                        #print(f"Added {len(pos_pairs)} pos pairs.")
                        geom2_coords = c(geometries2[idx2])
                        geom2_coords[:lengths2[idx2]] = geometry_format_convert(geom2_shifted, geometry_type2)
                        pos_geom1_coords.append(torch.tensor(geometries1[idx1], dtype=torch.float32))
                        pos_geom2_coords.append(torch.tensor(geom2_coords, dtype=torch.float32))
                        pos_lengths1.append(lengths1[idx1])
                        pos_lengths2.append(lengths2[idx2])

    X_geom1 = torch.cat([torch.stack(pos_geom1_coords), torch.stack(neg_geom1_coords)], dim=0)
    X_geom1_lengths = torch.cat([torch.tensor(pos_lengths1), torch.tensor(neg_lengths1)], dim=0)
    X_geom2 = torch.cat([torch.stack(pos_geom2_coords), torch.stack(neg_geom2_coords)], dim=0)
    X_geom2_lengths = torch.cat([torch.tensor(pos_lengths2), torch.tensor(neg_lengths2)], dim=0)

    Y = torch.cat(
        [torch.tensor([1] * num_pairs, dtype=torch.float32), torch.tensor([0] * num_pairs, dtype=torch.float32)], dim=0)
    torch.save((X_geom1, X_geom2, X_geom1_lengths, X_geom2_lengths, Y),
               f"{root_dir}/{dataset_name}/{geometry_type1}_{geometry_type2}_{relationship}_data.pt")
    pickle.dump((pos_pairs, neg_pairs),
                open(f"{root_dir}/{dataset_name}/{geometry_type1}_{geometry_type2}_{relationship}_index.pkl", "wb"))


def check_topological_relationship(geom1, geom2, geometry_type1, geometry_type2, relationship):
    # "intersect" includes "cover", "contain", "overlap", "touch", "equal"
    if relationship == "intersect":
        return geom1.intersects(geom2) and not geom1.equals(geom2) and not geom1.touches(geom2)
    elif relationship == "disjoint":
        return geom1.disjoint(geom2)
    # "cover" includes "equal"
    elif relationship == "cover":
        return geom1.covers(geom2) and not geom1.equals(geom2)
    elif relationship == "covered_by":
        return geom1.covered_by(geom2) and not geom1.equals(geom2)
    elif relationship == "overlap":
        return geom1.overlaps(geom2)
    # "contain" includes "equal"
    elif relationship == "contain":
        return geom1.contains(geom2) and not geom1.equals(geom2)
    elif relationship == "within":
        return geom1.within(geom2) and not geom1.equals(geom2)
    elif relationship == "touch":
        return geom1.touches(geom2)
    elif relationship == "equal":
        return geom1.equals(geom2)

    print(f"Unsupported relationship type: {relationship} between {geometry_type1} and {geometry_type2}")


def pairwise_topological_relationship(dataset_name, geometry_type1, geometry_type2, relationship_list, num_pairs=5000,
                                      max_attempts=10):
    geometries1, lengths1, num_geometries1 = load_dataset(dataset_name, geometry_type1)
    geometries2, lengths2, num_geometries2 = load_dataset(dataset_name, geometry_type2)
    print(
        f"Find {num_pairs} of {relationship_list} pairs from {num_geometries1} {geometry_type1} and {num_geometries2} {geometry_type2}.")

    result_pairs = []
    result_geom1_coords = []
    result_geom2_coords = []
    result_lengths1 = []
    result_lengths2 = []
    for relationship in relationship_list:
        print(relationship)
        pairs_indices = []
        temp_geom1_coords = []
        temp_geom2_coords = []
        temp_lengths1 = []
        temp_lengths2 = []
        while len(pairs_indices) < num_pairs:
            idx1 = random.randint(0, min(num_pairs, num_geometries1) - 1)
            idx2 = random.randint(0, min(num_pairs, num_geometries2) - 1)
            if [idx1, idx2] not in pairs_indices and [idx2, idx1] not in pairs_indices:
                geom1 = list2shapely(geometries1[idx1], lengths1[idx1], geometry_type1)
                geom2 = list2shapely(geometries2[idx2], lengths2[idx2], geometry_type2)
                if check_topological_relationship(geom1, geom2, geometry_type1, geometry_type2, relationship):
                    pairs_indices.append([idx1, idx2])
                    #print(f"Added {len(pairs_indices)} {relationship} pairs.")
                    temp_geom1_coords.append(torch.tensor(geometries1[idx1], dtype=torch.float32))
                    temp_geom2_coords.append(torch.tensor(geometries2[idx2], dtype=torch.float32))
                    temp_lengths1.append(lengths1[idx1])
                    temp_lengths2.append(lengths2[idx2])
                else:
                    # Normally it's not easy to find all relationships from the original dataset.
                    # We'll translate/scale the pair of geometries to satisfy the given relationship
                    if relationship == "touch":
                        point1, point2 = nearest_points(geom1, geom2)
                        shift_x = point1.x - point2.x
                        shift_y = point1.y - point2.y
                        geom2_shifted = translate(geom2, xoff=shift_x, yoff=shift_y)
                        if check_topological_relationship(geom1, geom2_shifted, geometry_type1, geometry_type2,
                                                          relationship):
                            pairs_indices.append([idx1, idx2])
                            #print(f"Added {len(pairs_indices)} {relationship} pairs.")
                            geom2_coords = c(geometries2[idx2])
                            geom2_coords[:lengths2[idx2]] = geometry_format_convert(geom2_shifted, geometry_type2)
                            temp_geom1_coords.append(torch.tensor(geometries1[idx1], dtype=torch.float32))
                            temp_geom2_coords.append(torch.tensor(geom2_coords, dtype=torch.float32))
                            temp_lengths1.append(lengths1[idx1])
                            temp_lengths2.append(lengths2[idx2])
                    elif relationship == "overlap" or relationship == "intersect":
                        point1, point2 = nearest_points(geom1, geom2)
                        shift_x = point1.x - point2.x
                        shift_y = point1.y - point2.y
                        geom2_shifted = translate(geom2, xoff=shift_x, yoff=shift_y)
                        # "intersect" includes more than "touch", so we shift the geom2_shifted further.
                        rand_factor = random.uniform(0, 1)
                        adjustment_x, adjustment_y = (geom1.centroid.x - geom2_shifted.centroid.x) * rand_factor, (
                                    geom1.centroid.y - geom2_shifted.centroid.y) * rand_factor  # Small adjustment factor
                        geom2_shifted = translate(geom2_shifted, xoff=adjustment_x, yoff=adjustment_y)
                        if check_topological_relationship(geom1, geom2_shifted, geometry_type1, geometry_type2,
                                                          relationship):
                            pairs_indices.append([idx1, idx2])
                            #print(f"Added {len(pairs_indices)} {relationship} pairs.")
                            geom2_coords = c(geometries2[idx2])
                            geom2_coords[:lengths2[idx2]] = geometry_format_convert(geom2_shifted, geometry_type2)
                            temp_geom1_coords.append(torch.tensor(geometries1[idx1], dtype=torch.float32))
                            temp_geom2_coords.append(torch.tensor(geom2_coords, dtype=torch.float32))
                            temp_lengths1.append(lengths1[idx1])
                            temp_lengths2.append(lengths2[idx2])

                    elif relationship in ["cover", "covered_by", "contain", "within"]:
                        if geometry_type1 == "polygon" and geometry_type2 == "polygon":
                            scale_factor = geom2.area / geom1.area
                        else:
                            scale_factor = 1
                        scale_factor = scale_factor * 5
                        geom2_scaled = scale(geom2, xfact=1 / scale_factor, yfact=1 / scale_factor, origin='center')
                        point1, point2 = nearest_points(geom1, geom2_scaled)
                        shift_x = point1.x - point2.x
                        shift_y = point1.y - point2.y
                        geom2_scaled = translate(geom2_scaled, xoff=shift_x, yoff=shift_y)
                        # Now geom2_scaled is supposed to be in proper size and "touch" geom1
                        # we only need to shift geom2_scaled
                        geom2_shifted = geom2_scaled
                        attempt = 1
                        while not check_topological_relationship(geom1, geom2_shifted, geometry_type1, geometry_type2,
                                                                 relationship) and attempt < max_attempts:
                            adjustment_x, adjustment_y = (geom1.centroid.x - geom2_scaled.centroid.x) * attempt / 10, (
                                    geom1.centroid.y - geom2_scaled.centroid.y) * attempt / 10  # Small adjustment factor
                            geom2_shifted = translate(geom2_scaled, xoff=adjustment_x, yoff=adjustment_y)
                            attempt += 1
                            # Check if intersected after adjustment
                        if check_topological_relationship(geom1, geom2_shifted, geometry_type1, geometry_type2,
                                                          relationship):
                            pairs_indices.append([idx1, idx2])
                            #print(f"Added {len(pairs_indices)} {relationship} pairs.")
                            geom2_coords = c(geometries2[idx2])
                            geom2_coords[:lengths2[idx2]] = geometry_format_convert(geom2_shifted, geometry_type2)
                            temp_geom1_coords.append(torch.tensor(geometries1[idx1], dtype=torch.float32))
                            temp_geom2_coords.append(torch.tensor(geom2_coords, dtype=torch.float32))
                            temp_lengths1.append(lengths1[idx1])
                            temp_lengths2.append(lengths2[idx2])

                    elif relationship == "equal":
                        pairs_indices.append([idx1, idx1])
                        #print(f"Added {len(pairs_indices)} {relationship} pairs.")
                        temp_geom1_coords.append(torch.tensor(geometries1[idx1], dtype=torch.float32))
                        temp_geom2_coords.append(torch.tensor(geometries2[idx1], dtype=torch.float32))
                        temp_lengths1.append(lengths1[idx1])
                        temp_lengths2.append(lengths2[idx1])

        result_pairs.extend(pairs_indices)
        result_geom1_coords.append(torch.stack(temp_geom1_coords))
        result_geom2_coords.append(torch.stack(temp_geom2_coords))
        result_lengths1.append(torch.tensor(temp_lengths1))
        result_lengths2.append(torch.tensor(temp_lengths2))

    X_geom1 = torch.cat(result_geom1_coords, dim=0)
    X_geom1_lengths = torch.cat(result_lengths1, dim=0)
    X_geom2 = torch.cat(result_geom2_coords, dim=0)
    X_geom2_lengths = torch.cat(result_lengths2, dim=0)

    Y = torch.cat([torch.tensor([i] * num_pairs, dtype=torch.float32) for i in range(len(relationship_list))], dim=0)
    torch.save((X_geom1, X_geom2, X_geom1_lengths, X_geom2_lengths, Y),
               f"{root_dir}/{dataset_name}/{geometry_type1}_{geometry_type2}_topological_relationship_data.pt")
    pickle.dump(result_pairs,
                open(f"{root_dir}/{dataset_name}/{geometry_type1}_{geometry_type2}_topological_relationship_index.pkl",
                     "wb"))



def check_directional_relationship_360(geom1, geom2):
    geom1_centroid = geom1.centroid
    geom2_centroid = geom2.centroid

    # Calculate relative differences
    dx = geom2_centroid.x - geom1_centroid.x
    dy = geom2_centroid.y - geom1_centroid.y
    # Calculate the angle in radians and convert to degrees
    angle = math.atan2(dy, dx)  # atan2 handles the signs of dx and dy correctly
    angle_degrees = math.degrees(angle)

    # Adjust the angle to be in the range [0, 360)
    if angle_degrees < 0:
        angle_degrees += 360

    group_size = 360 / 16
    if angle_degrees >= 360 - group_size / 2 or angle_degrees < group_size / 2:
        return str(0)

    return str(int((angle_degrees - group_size / 2) // group_size) + 1)


def pairwise_directional_relationship_360(dataset_name, geometry_type1, geometry_type2, relationship_list, num_pairs=5000):
    geometries1, lengths1, num_geometries1 = load_dataset(dataset_name, geometry_type1)
    geometries2, lengths2, num_geometries2 = load_dataset(dataset_name, geometry_type2)
    print(
        f"Find {num_pairs} of multiple direction pairs from {num_geometries1} {geometry_type1} and {num_geometries2} {geometry_type2}.")

    result_pairs = []
    result_geom1_coords = []
    result_geom2_coords = []
    result_lengths1 = []
    result_lengths2 = []
    for relationship in relationship_list:
        print(relationship)
        pairs_indices = []
        temp_geom1_coords = []
        temp_geom2_coords = []
        temp_lengths1 = []
        temp_lengths2 = []
        while len(pairs_indices) < num_pairs:
            idx1 = random.randint(0, min(num_pairs, num_geometries1) - 1)
            idx2 = random.randint(0, min(num_pairs, num_geometries2) - 1)
            geom1 = list2shapely(geometries1[idx1], lengths1[idx1], geometry_type1)
            geom2 = list2shapely(geometries2[idx2], lengths2[idx2], geometry_type2)
            # if check_directional_relationship(geom1, geom2) == relationship and [idx1, idx2] not in pairs_indices:
            if check_directional_relationship_360(geom1, geom2) == relationship and [idx1, idx2] not in pairs_indices:
                pairs_indices.append([idx1, idx2])
                #print(f"Added {len(pairs_indices)} {relationship} pairs.")
                temp_geom1_coords.append(torch.tensor(geometries1[idx1], dtype=torch.float32))
                temp_geom2_coords.append(torch.tensor(geometries2[idx2], dtype=torch.float32))
                temp_lengths1.append(lengths1[idx1])
                temp_lengths2.append(lengths2[idx2])
        result_pairs.extend(pairs_indices)
        result_geom1_coords.append(torch.stack(temp_geom1_coords))
        result_geom2_coords.append(torch.stack(temp_geom2_coords))
        result_lengths1.append(torch.tensor(temp_lengths1))
        result_lengths2.append(torch.tensor(temp_lengths2))

    X_geom1 = torch.cat(result_geom1_coords, dim=0)
    X_geom1_lengths = torch.cat(result_lengths1, dim=0)
    X_geom2 = torch.cat(result_geom2_coords, dim=0)
    X_geom2_lengths = torch.cat(result_lengths2, dim=0)

    Y = torch.cat([torch.tensor([i] * num_pairs, dtype=torch.float32) for i in range(len(relationship_list))], dim=0)
    torch.save((X_geom1, X_geom2, X_geom1_lengths, X_geom2_lengths, Y),
               f"{root_dir}/{dataset_name}/{geometry_type1}_{geometry_type2}_directional_relationship_360_data.pt")
    pickle.dump(result_pairs,
                open(f"{root_dir}/{dataset_name}/{geometry_type1}_{geometry_type2}_directional_relationship_360_index.pkl",
                     "wb"))


def pairwise_distance(dataset_name, geometry_type1, geometry_type2, num_pairs=10000):
    from exp_knn import generate_pairwise_ground_truth_distance
    geometries1, lengths1, num_geometries1 = load_dataset(dataset_name, geometry_type1)
    geometries2, lengths2, num_geometries2 = load_dataset(dataset_name, geometry_type2)

    indexes1 = []
    indexes2 = []
    pairs = []
    while len(indexes1) < num_pairs:
        idx1 = random.randint(0, min(num_geometries1, num_pairs) - 1)
        idx2 = random.randint(0, min(num_geometries2, num_pairs) - 1)
        while idx2 == idx1:
            idx2 = random.randint(0, min(num_geometries2, num_pairs) - 1)
        if [idx1, idx2] not in pairs:
            indexes1.append(idx1)
            indexes2.append(idx2)
            pairs.append([idx1, idx2])

    geom_data1 = torch.tensor(geometries1, dtype=torch.float32)[indexes1]
    geom_data2 = torch.tensor(geometries2, dtype=torch.float32)[indexes2]
    dis_matrix = generate_pairwise_ground_truth_distance(geom_data1, geom_data2, object_type=geometry_type2)
    torch.save((geom_data1, geom_data2, torch.tensor(lengths1)[indexes1], torch.tensor(lengths2)[indexes2], dis_matrix),
               f"{root_dir}/{dataset_name}/{geometry_type1}_{geometry_type2}_dis_data.pt")
    pickle.dump((indexes1, indexes2),
                open(f"{root_dir}/{dataset_name}/{geometry_type1}_{geometry_type2}_dis_index.pt", "wb"))


root_dir = "./Poly2Vec/data"
if __name__ == "__main__":
    for dataset_name in ["Singapore", "NewYork"]:
        print(dataset_name)
        # Topological
        print("Topological relationship")
        for geometry_type1, geometry_type2, relationship_list in [["polygon", "polygon", ["disjoint", "touch", "overlap", "contain", "within", "equal"]],
                                                                  ["polygon", "polyline", ["disjoint", "touch", "intersect", "contain"]]]:
            print(geometry_type1, geometry_type2)
            pairwise_topological_relationship(dataset_name, geometry_type1, geometry_type2, relationship_list=relationship_list, num_pairs=5000, max_attempts=10)
        for geometry_type1, geometry_type2, relationship in [["polygon", "point", "contain"], ["polyline", "polyline", "intersect"], ["polyline", "point", "contain"]]:
            print(geometry_type1, geometry_type2)
            pairwise_relationship(dataset_name, geometry_type1, geometry_type2, relationship)
        # Directional
        print("Directional relationship")
        for geometry_type1, geometry_type2 in [["polygon", "polygon"], ["polygon", "polyline"], ["polygon", "point"], ["polyline", "polyline"], ["polyline", "point"], ["point", "point"]]:
            pairwise_directional_relationship_360(dataset_name, geometry_type1, geometry_type2, [str(i) for i in range(16)], num_pairs=5000)
        # Distance
        print("Distance relationship")
        for geometry_type1, geometry_type2 in [["point", "point"], ["point", "polyline"], ["point", "polygon"]]:
            pairwise_distance(dataset_name, geometry_type1, geometry_type2, num_pairs=10000)

