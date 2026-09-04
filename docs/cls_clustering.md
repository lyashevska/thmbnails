# CLS embedding clustering: methods and results

Short write-up for the thumbnail-level (CLS) density clustering. Figures to inspect: `data/dinov3_cls_clusters/nopca-n15-mcs20-ms20/umap.png` and `samples/cluster_*/_grid.jpg`; peel grids under `nopca-n15-mcs20-ms20-r1/` and `-r2/`. Run log: [dinov3_runs.md](dinov3_runs.md).

## Methods

### Corpus and embeddings

The clustering corpus is the set of year-stratified video thumbnails that pass the same validity filter used elsewhere in the study (readable JPEG, source size 640×360, file size ≥ 4 KB). Of 12,500 sampled videos (2,500 per year, 2020–2024), 8,666 thumbnails meet these criteria.

Each valid thumbnail is converted to RGB, letterboxed to a square with black padding (preserving 16:9 composition), and resized to 224×224 pixels. A DINOv3 ViT-L/16 model pretrained on LVD-1689M yields a 1024-dimensional CLS vector per image. Inference is deterministic given the checkpoint; vectors are stored as an \(N \times 1024\) matrix with a matching image-id list. The production matrix has shape \((8666, 1024)\).

### Dimensionality reduction and clustering

Density clustering is performed in a 10-dimensional UMAP embedding of the raw 1024-D CLS vectors (cosine metric, `n_neighbors=15`, `min_dist=0`, random state 42). There is no PCA step. A separate two-dimensional UMAP with the same neighbourhood parameters is used only for visualisation, not for assigning labels.

HDBSCAN is fit in the 10-D UMAP with Euclidean distance, excess-of-mass cluster selection, `min_cluster_size=20`, and `min_samples=20`. Points not assigned to a cluster are labelled noise (\(-1\)) and are treated as unstructured. In addition, after inspecting sample montages, oversized mixed groups (a large share of a round’s mass, visually heterogeneous) are treated as **residual / unstructured clusters**, not as thumbnail types. This configuration is the working cut (`nopca-n15-mcs20-ms20`).

HDBSCAN on 50-D PCA, PCA then UMAP then HDBSCAN, and \(K=40\) k-means on PCA were also run. They are not the operating typology; see [Other options considered](#other-options-considered).

### Parameter selection

UMAP neighbourhood size (`n_neighbors` ∈ {15, 30, 50}), UMAP `min_dist` ∈ {0.0, 0.1, 0.25}, and HDBSCAN `min_cluster_size` ∈ {10, 20, 30, 50} and `min_samples` ∈ {5, 10, \(\min\_cluster\_size\)} were crossed (99 unique cells after dropping duplicate `min_samples` pairs). Each cell used raw CLS as the UMAP source, a 10-D UMAP clustering space, and excess-of-mass HDBSCAN. Two-dimensional plots and thumbnail grids were not generated during the sweep (`sweeps/nopca`).

Cells were scored with a composite of density-based cluster validity (DBCV, weight 0.5), silhouette on non-noise points (0.3), and one minus the noise fraction (0.2), each min–max normalised over the grid. The composite was used only to rank the grid. The operating configuration was chosen among cells with a reviewable number of clusters (approximately 20–80), non-trivial leftover mass, and median cluster size above `min_cluster_size`, then confirmed by inspecting sample montages. Visual coherence of those montages is treated as the primary validity criterion.

### Stability

After freezing the working cut, UMAP was refit 100 times with distinct random seeds (raw CLS held fixed; HDBSCAN knobs unchanged). Agreement among the 100 partitions was summarised as pairwise adjusted Rand index (ARI) and normalised mutual information (NMI), mean ± standard deviation over 4,950 pairs. Noise was treated as its own label in the primary scores; a second pair of scores used only images clustered in both members of a pair. The same 100-seed protocol on cut A (PCA then 10-D UMAP, same HDBSCAN knobs) is the comparison (`sweep-A-stability`).

### Residual clustering (noise peel)

HDBSCAN noise is not discarded as unstructured. Thumbnails labelled \(-1\) under the working cut were extracted and the pipeline was **refit** on that subset only (new 10-D UMAP on raw CLS, same HDBSCAN knobs). A second peel repeated the procedure on remaining noise. Cluster identifiers in the combined table are offset so that round-0, round-1, and round-2 labels do not collide. Coordinates from later rounds are not plotted on the original UMAP.

## Results

### Chosen cut

The composite ranking was dominated by two-cluster, zero-noise splits of the UMAP cloud (median cluster size 4,333). Those solutions were discarded as a bipartition of the embedding, not a taxonomy of thumbnail types. Twelve cells fell in the usable band (20–80 clusters); almost all of them had `min_dist=0`. `min_dist` of 0.1 and 0.25 collapsed the map.

The working cut (`n_neighbors=15`, `min_dist=0`, `min_cluster_size=20`, `min_samples=20`) produced **33 clusters** and **41.1% noise** (3,558 / 8,666 images) on seed 42. Median cluster size was 54 (range 22–842). DBCV was 0.27 and silhouette 0.54. Clusters 0 and 18 (741 and 842 images) are residual rather than types; most other groups were in the tens to low hundreds. Relative to a finer cut on the same UMAP (`min_cluster_size=10`, `min_samples=10`: 71 clusters, 43.2% noise, median size 31), the working cut was preferred for tighter cores and a reviewable number of groups, with leftover mass for residual clustering.

### Stability

Over 100 UMAP seeds, mean pairwise ARI was **0.47 (SD 0.35)** and NMI **0.57 (SD 0.24)**. Restricting to points clustered in both runs did not change the picture (ARI 0.53, SD 0.42; NMI 0.66, SD 0.27). Cut A, with the same knobs after a PCA step, was indistinguishable on this protocol (ARI 0.45, SD 0.37; NMI 0.55, SD 0.29; assigned-only ARI 0.50, NMI 0.61). The large standard deviations indicate a mixture of working-cut-like partitions and collapsed maps rather than a tight plateau. Dropping PCA did not stabilise the typology. The working cut is therefore reported as an **exploratory, seed-42 density clustering**, not as a unique 33-type taxonomy. Where a seed-invariant labelling of every thumbnail is required, the \(K=40\) k-means partition on PCA is used instead.

| Probe | ARI | NMI | Assigned-only ARI | Assigned-only NMI |
|-------|-----|-----|-------------------|-------------------|
| Working cut, 100 seeds (`nopca-n15-mcs20-ms20-stability`) | 0.47 ± 0.35 | 0.57 ± 0.24 | 0.53 ± 0.42 | 0.66 ± 0.27 |
| Cut A, 100 seeds (`sweep-A-stability`) | 0.45 ± 0.37 | 0.55 ± 0.29 | 0.50 ± 0.42 | 0.61 ± 0.31 |

### Noise peels

Refitting the pipeline on the 3,558 unclustered thumbnails yielded **11 additional clusters** and 1,254 remaining noise (35.2% of the leftover set). DBCV and silhouette on this subset were low (0.07 and 0.13). Peel cluster 3 (combined id 36; 1,402 images) is the mixed residual pile for that round. The other 10 groups were smaller (29–401 images) and are candidates for visual review only.

A second peel on the remaining 1,254 images yielded **9 clusters** and 632 noise (50.4% of that subset; **7.3% of the full corpus**). DBCV and silhouette were higher than in the first peel (0.23 and 0.45). No single group took a large share of the round (largest 160 and 145). Further peels were not run.

Combining offset labels across rounds gives 53 cluster identifiers and 8,034 assigned thumbnails (92.7%). Round 0 remains the primary taxonomy (5,108 images, 33 groups). Rounds 1 and 2 are secondary looks at leftovers (2,304 and 622 images) and are not given equal interpretive weight. Sample montages for peel clusters—not the combined UMAP, whose coordinates mix incompatible maps—are the basis for accepting or rejecting those extra groups.

### Unstructured labels

HDBSCAN noise (\(-1\)) is unstructured by construction. The following **assigned** clusters are also treated as unstructured residual mass (size plus mixed montages), not as named thumbnail types. Folder ids are those in `samples/cluster_<id>/`; combined ids are those in `nopca-n15-mcs20-ms20-peels/combined_assignments.csv`.

| Round | Run folder | Folder id | Combined id | n | Share |
|-------|------------|----------:|------------:|--:|-------|
| 0 | `nopca-n15-mcs20-ms20` | −1 (noise) | −1 | 3,558 then peeled | 41.1% of corpus before peels |
| 0 | same | **0** | **0** | 741 | 8.6% of corpus |
| 0 | same | **18** | **18** | 842 | 9.7% of corpus |
| 1 | `nopca-n15-mcs20-ms20-r1` | **3** | **36** | 1,402 | 39.4% of r1 input; 16.2% of corpus |
| 1 | same | −1 (noise) | (peeled in r2) | 1,254 | 35.2% of r1 input |
| 2 | `nopca-n15-mcs20-ms20-r2` | −1 (noise) | **−1** | 632 | 7.3% of corpus (final leftover) |

After two peels, final unstructured mass is: remaining noise 632; residual clusters 0, 18, and 36 (741 + 842 + 1,402 = 2,985); **3,617 / 8,666 images (41.7%)** if those residual clusters are counted with noise. Interpretive analysis of visual types should use the other round-0 clusters (and, where montages support it, r1 folders other than 3 and the r2 folders).

### Other options considered

Three alternatives were fit on the same embeddings and are not used as the operating typology.

- **HDBSCAN in 50-D PCA** (`hdbscan-eom-vitl`): 539 clusters, 69.2% noise, mostly size-3 groups. Density clustering in that Euclidean space did not yield a usable thumbnail typology, which is why UMAP is used to reshape neighbourhoods before HDBSCAN.
- **PCA (50 components, 59% variance) then 10-D UMAP then HDBSCAN**, independently tuned on the same 99-cell grid (`sweep-A-n15-mcs20-ms20`): the usable-band winner used the **same knobs** as the working cut (36 clusters, 38.6% noise). Assigned-only ARI versus the no-PCA cut was 0.90. Native UMAP validity was slightly higher with PCA; silhouette in the original CLS space was slightly higher without it. The extra linear reduction was dropped as unnecessary for the primary cut.
- **\(K=40\) k-means on PCA** (`kmeans-k40-vitl`): a deterministic, 0% noise partition of the full corpus, used when every thumbnail must receive a label.

### Interpretation for the study

Density clustering of DINOv3 CLS embeddings recovers a moderate number of recurring thumbnail regimes without using titles or VLM labels. The first-pass cut leaves a large unclustered remainder by design; successive peels show that some additional local structure exists among leftovers, alongside large mixed residuals. Because UMAP+HDBSCAN partitions are seed-sensitive on this corpus, claims about specific visual types should be grounded in the seed-42 montages of the working cut (and, where grids support it, in the smaller peel folders), and should not be stated as a unique or fully reproducible typology of the 8,666 images.
