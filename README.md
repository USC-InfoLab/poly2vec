# Poly2Vec

**Poly2Vec** is a unified encoding framework that encodes points, polylines, and polygons into a fixed-length representation for spatial reasoning tasks.
---

## 📰 Paper

Our approach is presented in the following paper accepted to **ICML 2025**:

> **Poly2Vec: Polymorphic Fourier-Based Encoding of Geospatial Objects for GeoAI Applications**  
> *Maria Despoina Siampou, Jialiang Li, John Krumm, Cyrus Shahabi, Hua Lu*  
> [📄 Check out the paper](https://arxiv.org/pdf/2408.14806)

## 🔧 Installation

We recommend using a Conda environment with Python ≥ 3.9:

```bash
conda create -n poly2vec python=3.9
conda activate poly2vec
```
Then install the dependencies:

```bash
pip install -r requirements.txt
```

## 📦 Preprocessed Data

We utilized two OpenStreetMap (OSM) datasets in our evaluation: **Singapore** and **New York**. The required data can be downloaded from [Geofabrik](https://download.geofabrik.de/).

### OSM Tags Used
- **Points (POIs)**: `amenity`, `shop`, `tourism`, `leisure`
- **Polylines (Roads)**: `motorway`, `trunk`, `primary`
- **Polygons (Buildings)**: `building`

### Preprocessing Steps

1. Place the downloaded `.osm.pbf` files into the `./data/` directory.

2. Run the preprocessing script to extract and normalize geometries:

```bash
python utils/data_preprocessing.py
```

his will generate the following files inside each dataset folder:

- `poi_normalized.pkl`
- `roads_normalized.pkl`
- `buildings_normalized.pkl`

### Generate Datasets for Training and Evaluation

Run the following script:

```bash
python utils/data_generation.py
```

You will get files like `polygon_polygon_topological_relationship_data.pt` in each dataset's folder.

### Link to Datasets
For completeness, we also provide a link to the preprocessed data used for all the experiments (including RegionDCL) 
```
https://drive.google.com/drive/folders/119KvtA1K9CbII6TMMSPXkd1Yjrxhv9-P?usp=sharing
```

## 🚀 Training
Specify your training setup in the `config.json` file.

### Example Run
To train a model on the New York polygon-polygon topological relation dataset using Poly2Vec:

```bash
python run.py \
  -dataset_name "NewYork" \
  -dataset_type1 "polygons" \
  -dataset_type2 "polygons" \
  -task "multi-relation" \
  -data_file "./data/NewYork/polygon_polygon_intersect_data.pt" \
  -encoder_type "poly2vec" \
  -data_path "./data/NewYork" \
  -num_classes 6
```

## 📄 Citation
If you found `Poly2Vec` useful, please consider citing us:

```bibtex
@inproceedings{siampoupoly2vec,
  title={Poly2Vec: Polymorphic Fourier-Based Encoding of Geospatial Objects for GeoAI Applications},
  author={Siampou, Maria Despoina and Li, Jialiang and Krumm, John and Shahabi, Cyrus and Lu, Hua},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```
