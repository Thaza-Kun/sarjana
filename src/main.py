from workflows.replications import bo_han_chen_2021
from workflows.experiments import UMAP_HDBSCAN_FRBSTATS

import logging
from utils import (
    datalogger,
    flowlogger,
)


datalogger.setLevel(logging.DEBUG)
flowlogger.setLevel(logging.DEBUG)

if __name__ == "__main__":
    seed = 42
    size = 19
    # for size in range(2, 50):
    bo_han_chen_2021(min_cluster_size=size, seed=seed, debug=True)
    UMAP_HDBSCAN_FRBSTATS(min_cluster_size=size, seed=seed, debug=True)
