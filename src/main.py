import workflows

if __name__ == "__main__":
    seed = 42
    min_cluster_size = 19
    workflows.replicate_chen2021_UMAP_HDBSCAN(
        min_cluster_size=min_cluster_size, seed=seed
    )
