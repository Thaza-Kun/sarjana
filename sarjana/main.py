from workflows.replications import bo_han_chen_2021
from workflows.experiments import UMAP_HDBSCAN_FRBSTATS

import logging
from sarjana.utils.logger import datalogger, flowlogger
from sarjana.preamble import ExecutionOptions

if __name__ == "__main__":
    ExecutionOptions.Mode = "debug"
    seed = 42
    size = 19
    # for size in range(2, 50):
    bo_han_chen_2021(min_cluster_size=size, seed=seed, debug=True)
    UMAP_HDBSCAN_FRBSTATS(min_cluster_size=size, seed=seed, debug=True)

# TODO Add preamble module. (For setup, configs, etc.)
# Using the preamble, add ability to switch output to non-tracked files and tracked files
# TODO Add testing?
