from sarjana.preamble import ExecutionOptions

ExecutionOptions.Mode = "debug"

from workflows.replications import bo_han_chen_2021
from workflows.experiments import UMAP_HDBSCAN_FRBSTATS, HDBSCAN_important_features

if __name__ == "__main__":
    seed = 42
    size = 19
    # for size in range(2, 50):
    #     # bo_han_chen_2021(min_cluster_size=size, seed=seed, debug=True)
    #     UMAP_HDBSCAN_FRBSTATS(min_cluster_size=size, seed=seed, debug=True)
    data, result = HDBSCAN_important_features(min_cluster_size=size, seed=seed)
    print("score: {}".format(result))

# TODO Add testing?
# TODO Add Parameters
# TODO Add Feature Importance
