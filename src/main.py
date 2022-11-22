import workflows

if __name__ == "__main__":
    seed = 42
    cluster_size = 19
    workflows.replicate_chen2021_UMAP_HDBSCAN(min_cluster_size=cluster_size, seed=seed)
    workflows.UMAP_HDBSCAN_no_model_dependent_params(
        min_cluster_size=cluster_size, seed=seed
    )
