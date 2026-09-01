# CLS embedding clustering: methods and results

Short write-up for the thumbnail-level (CLS) density clustering. Figures to inspect: `data/dinov3_cls_clusters/sweep-A-n15-mcs20-ms20/umap.png` and `samples/cluster_*/_grid.jpg`; peel grids under `sweep-A-n15-mcs20-ms20-r1/` and `-r2/`. Run log: [dinov3_runs.md](dinov3_runs.md).

## Methods

### Corpus and embeddings

The clustering corpus is the set of year-stratified video thumbnails that pass the same validity filter used elsewhere in the study (readable JPEG, source size 640×360, file size ≥ 4 KB). Of 12,500 sampled videos (2,500 per year, 2020–2024), 8,666 thumbnails meet these criteria.

Each valid thumbnail is converted to RGB, letterboxed to a square with black padding (preserving 16:9 composition), and resized to 224×224 pixels. A DINOv3 ViT-L/16 model pretrained on LVD-1689M yields a 1024-dimensional CLS vector per image. Inference is deterministic given the checkpoint; vectors are stored as an \(N \times 1024\) matrix with a matching image-id list. The production matrix has shape \((8666, 1024)\).

### Dimensionality reduction and clustering

CLS vectors are reduced with PCA (50 components, 59.0% of variance retained, random state 42). Density clustering is performed in a 10-dimensional UMAP embedding of that PCA space (cosine metric, `n_neighbors=15`, `min_dist=0`, random state 42). A separate two-dimensional UMAP with the same neighbourhood parameters is used only for visualisation, not for assigning labels.

HDBSCAN is fit in the 10-D UMAP with Euclidean distance, excess-of-mass cluster selection, `min_cluster_size=20`, and `min_samples=20`. Points not assigned to a cluster are labelled noise (\(-1\)) and are treated as unstructured. In addition, after inspecting sample montages, oversized mixed groups (a large share of a round’s mass, visually heterogeneous) are treated as **residual / unstructured clusters**, not as thumbnail types. This configuration is referred to below as **cut A**.


### Parameter selection

UMAP neighbourhood size (`n_neighbors` ∈ {15, 30, 50}), UMAP `min_dist` ∈ {0.0, 0.1, 0.25}, and HDBSCAN `min_cluster_size` ∈ {10, 20, 30, 50} and `min_samples` ∈ {5, 10, \(\min\_cluster\_size\)} were crossed (99 unique cells after dropping duplicate `min_samples` pairs). Each cell used the same PCA, 10-D UMAP clustering space, and excess-of-mass HDBSCAN. Two-dimensional plots and thumbnail grids were not generated during the sweep.

Cells were scored with a composite of density-based cluster validity (DBCV, weight 0.5), silhouette on non-noise points (0.3), and one minus the noise fraction (0.2), each min–max normalised over the grid. The composite was used only to rank the grid. The operating configuration was chosen among cells with a reviewable number of clusters (approximately 20–80), non-trivial leftover mass, and median cluster size above `min_cluster_size`, then confirmed by inspecting sample montages. Visual coherence of those montages is treated as the primary validity criterion.

### Stability

After freezing cut A, UMAP was refit 100 times with distinct random seeds (PCA held fixed; HDBSCAN knobs unchanged). Agreement among the 100 partitions was summarised as pairwise adjusted Rand index (ARI) and normalised mutual information (NMI), mean ± standard deviation over 4,950 pairs. Noise was treated as its own label in the primary scores; a second pair of scores used only images clustered in both members of a pair. The same protocol was repeated for two alternative geometries (10-D UMAP with `n_neighbors=50`; 5-D UMAP with `n_neighbors=50`) at 10 seeds each, as a check that the instability was not an artefact of one UMAP setting.

### Residual clustering (noise peel)

HDBSCAN noise is not discarded as unstructured. Thumbnails labelled \(-1\) under cut A were extracted and the full pipeline was **refit** on that subset only (new PCA, new 10-D UMAP, same HDBSCAN knobs). A second peel repeated the procedure on remaining noise. Cluster identifiers in the combined table are offset so that round-0 (cut A), round-1, and round-2 labels do not collide. Coordinates from later rounds are not plotted on the original UMAP.

## Results

### Chosen cut (A)

The composite ranking was dominated by two-cluster, zero-noise splits of the UMAP cloud (median cluster size 4,333). Those solutions were discarded as a bipartition of the embedding, not a taxonomy of thumbnail types. Every sweep cell with 20 or more clusters had `min_dist=0`; `min_dist` of 0.1 and 0.25 collapsed the map.

Cut A (`n_neighbors=15`, `min_dist=0`, `min_cluster_size=20`, `min_samples=20`) produced **36 clusters** and **38.6% noise** (3,348 / 8,666 images) on seed 42. Median cluster size was 65 (range 20–834). DBCV was 0.32 and silhouette 0.59. Clusters 0 and 18 (738 and 834 images) are residual rather than types; most other groups were in the tens to low hundreds. Relative to a coarser cut on the same UMAP (`min_cluster_size=30`, `min_samples=10`: 32 clusters, 34.0% noise, median size 92), A was preferred for tighter cores and a larger leftover set for residual clustering.

### Stability

Over 100 UMAP seeds, mean pairwise ARI was **0.45 (SD 0.37)** and NMI **0.55 (SD 0.29)**. Restricting to points clustered in both runs did not change the picture (ARI 0.50, SD 0.42; NMI 0.61, SD 0.31). Alternative UMAP geometries did not improve agreement (`n_neighbors=50`, 10 seeds: ARI 0.51, SD 0.41; 5-D UMAP: ARI 0.51, SD 0.41). The large standard deviations indicate a mixture of A-like partitions and collapsed maps rather than a tight plateau. Cut A is therefore reported as an **exploratory, seed-42 density clustering**, not as a unique 36-type taxonomy. Where a seed-invariant labelling of every thumbnail is required, the \(K=40\) k-means partition on PCA is used instead.

### Noise peels

Refitting the pipeline on A’s 3,348 unclustered thumbnails yielded **18 additional clusters** and 1,096 remaining noise (32.7% of the leftover set). DBCV and silhouette on this subset were low (0.16 and 0.16). Peel cluster 7 (combined id 43; 1,231 images) is the mixed residual pile for that round. The other 17 groups were small (27–207 images) and are candidates for visual review only.

A second peel on the remaining 1,096 images yielded **6 clusters** and 316 noise (28.8% of that subset; **3.6% of the full corpus**). DBCV and silhouette were higher than in the first peel (0.36 and 0.38), consistent with a smaller leftover pool. Peel cluster 1 (combined id 55; 346 images) is the residual pile for that round. Further peels were not run.

Combining offset labels across rounds gives 60 cluster identifiers and 8,350 assigned thumbnails (96.4%). Round 0 (cut A) remains the primary taxonomy (5,318 images, 36 groups). Rounds 1 and 2 are secondary looks at leftovers (2,252 and 780 images) and are not given equal interpretive weight. Sample montages for peel clusters—not the combined UMAP, whose coordinates mix incompatible maps—are the basis for accepting or rejecting those extra groups.

### Unstructured labels

HDBSCAN noise (\(-1\)) is unstructured by construction. The following **assigned** clusters are also treated as unstructured residual mass (size plus mixed montages), not as named thumbnail types. Folder ids are those in `samples/cluster_<id>/`; combined ids are those in `sweep-A-n15-mcs20-ms20-peels/combined_assignments.csv`.

| Round | Run folder | Folder id | Combined id | n | Share |
|-------|------------|----------:|------------:|--:|-------|
| 0 | `sweep-A-n15-mcs20-ms20` | −1 (noise) | −1 | 3,348 then peeled | 38.6% of corpus before peels |
| 0 | same | **0** | **0** | 738 | 8.5% of corpus |
| 0 | same | **18** | **18** | 834 | 9.6% of corpus |
| 1 | `sweep-A-n15-mcs20-ms20-r1` | **7** | **43** | 1,231 | 36.8% of r1 input; 14.2% of corpus |
| 1 | same | −1 (noise) | (peeled in r2) | 1,096 | 32.7% of r1 input |
| 2 | `sweep-A-n15-mcs20-ms20-r2` | **1** | **55** | 346 | 31.6% of r2 input; 4.0% of corpus |
| 2 | same | −1 (noise) | **−1** | 316 | 3.6% of corpus (final leftover) |

After two peels, final unstructured mass is: remaining noise 316; residual clusters 0, 18, 43, and 55 (738 + 834 + 1,231 + 346 = 3,149); **3,465 / 8,666 images (40.0%)** if those residual clusters are counted with noise. Interpretive analysis of visual types should use the other A clusters (and, where montages support it, r1 folders other than 7 and r2 folders other than 1).

### Interpretation for the study

Density clustering of DINOv3 CLS embeddings recovers a moderate number of recurring thumbnail regimes without using titles or VLM labels. The first-pass cut leaves a large unclustered remainder by design; successive peels show that some additional local structure exists among leftovers, alongside large mixed residuals. Because UMAP+HDBSCAN partitions are seed-sensitive on this corpus, claims about specific visual types should be grounded in the seed-42 montages of cut A (and, where grids support it, in the smaller peel folders), and should not be stated as a unique or fully reproducible typology of the 8,666 images.
