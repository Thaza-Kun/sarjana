import workflows

if __name__ == "__main__":
    seed = 42
    cluster_size = 19
    workflows.replicate_chen2021(min_cluster_size=cluster_size, seed=seed)
    workflows.replicate_chen2021_model_independent(
        min_cluster_size=cluster_size, seed=seed
    )
    workflows.UMAP_HDBSCAN_FRBSTATS(min_cluster_size=cluster_size, seed=seed)
