import logging
import workflows

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    seed = 42
    cluster_size = 19
    # workflows.replicate_chen2021(min_cluster_size=cluster_size, seed=seed)
    # workflows.replicate_chen2021_model_independent(
    #     min_cluster_size=cluster_size, seed=seed
    # )
    workflows.UMAP_HDBSCAN_FRBSTATS(min_cluster_size=cluster_size, seed=seed)

# TODO split workflows file to workflows/replication.py, workflows/modified.py, workflows/experimental.py
