import argparse
from pathlib import Path

import geopandas as gpd
from pyiceberg.catalog import load_catalog
from pyiceberg.exceptions import NamespaceAlreadyExistsError
import pyarrow.parquet as pq


def build_tables(file: Path):
    file_dir = file.parent
    warehouse_path = Path("/tmp/warehouse")
    warehouse_path.mkdir(exist_ok=True)
    catalog_settings = {
        "type": "sql",
        "uri": f"sqlite:///{warehouse_path}/pyiceberg_catalog.db",
        "warehouse": f"file://{warehouse_path}",
    }

    namespace = "hydrofabric"
    catalog = load_catalog(namespace, **catalog_settings)
    try:
        catalog.create_namespace(namespace)
    except NamespaceAlreadyExistsError:
        print(f"Namespace {namespace} already exists")
    layers = [
        "divide-attributes",
        "divides",
        "flowpath-attributes-ml",
        "flowpath-attributes",
        "flowpaths",
        "hydrolocations",
        "lakes",
        "network",
        "nexus",
        "pois",
    ]
    for layer in layers:
        print(f"Building layer: {layer}")
        if catalog.table_exists(f"{namespace}.{layer}"):
            print(f"Table {layer} already exists. Skipping build")
        else:
            parquet_file = file_dir / f"{layer}.parquet"
            if parquet_file.exists() is False:
                print(f"Converting layer: {layer} to parquet")
                gdf = gpd.read_file(file, layer=layer)
                gdf.to_parquet(parquet_file)
            arrow_table = pq.read_table(parquet_file)
            iceberg_table = catalog.create_table(
                f"{namespace}.{layer}",
                schema=arrow_table.schema,
                location=str(warehouse_path / "data"),
            )
            iceberg_table.append(arrow_table)
    print(f"Build successful. Files written into metadata store @ {warehouse_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A module to convert a hydrofabric geopackage into a pyiceberg warehouse")
    parser.add_argument("--file", required=True, help="The hydrofabric geopackage to build a warehouse from")

    args = parser.parse_args()
    file = Path(args.file)
    if file.exists():
        build_tables(file=file)
    else:
        msg = f"File not found: {file}"
        print(msg)
        raise FileNotFoundError(msg)
    