import argparse
import json

def args_parser(json_file):
    parser = argparse.ArgumentParser(
        "Poly2Vec",
    )
    with open(json_file, 'r') as file:
        data = json.load(file)
    for key, value in data.items():
        parser.add_argument(f'--{key}', type=type(value), default=value, help=f'Description for {key}')
    args, _ = parser.parse_known_args()
    return args