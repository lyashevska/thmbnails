# Residual CLS (Stage 1): which clusters to keep

Operating Stage 1 cut: `nopca-n15-mcs20-ms20-residual-leaf` (HDBSCAN **leaf**, `min_samples=10`; 68 residual clusters, 56.2% noise). Parent cut: `nopca-n15-mcs20-ms20` (33 clusters). Methods: [cls_clustering.md](cls_clustering.md). Run log: [dinov3_runs.md](dinov3_runs.md).

Figures: `data/dinov3_cls_clusters/nopca-n15-mcs20-ms20-residual-leaf/samples/cluster_*/_grid.jpg`. Mix table: `parent_residual_mix.csv`. Each assigned thumbnail is `(parent_cluster_id, cluster_id)` — parent scene type plus residual look.

The inherited-knob run (`nopca-n15-mcs20-ms20-residual/`, eom, `min_samples=20`) is a control only and is not used. It is a two-cluster split of the residual map (illustrated vs the rest), not a leftover taxonomy.

## Criterion

Mix numbers come from `parent_residual_mix.csv`. Visual claims are grounded in the 12-up sample montages, not in the full cluster membership. Residual ids are folder ids under `samples/cluster_<id>/`.

Parent ids used below:

| Parent | n in parent cut | Look |
|--------|----------------:|------|
| 0 | 741 | Illustrated / drawing / 3D (unstructured residual in the parent cut) |
| 14 | 348 | Gay male |
| 15 | 432 | Hetero couple, studio |
| 17 | 411 | Hetero couple, bed / amateur-euro |
| 18 | 842 | Close-up oral / genital (unstructured residual in the parent cut) |
| 22 | 111 | Genital close-up |
| 27 | 268 | Lingerie / stockings |
| 28 | 180 | Breast close-up |
| 32 | 313 | Rear / from-behind |

Parent −1 is HDBSCAN noise from the parent cut (3,558 images). Many useful residual groups have majority parent −1: they pulled leftover images into a type by leftover appearance.

## Keep

### Clothing and costume

| Residual | n | n parents | Majority | Shared look |
|----------|--:|----------:|----------|-------------|
| **24** | 103 | 17 | parent 27 (32%) | Fishnets, across sofa, couple, amateur, cam |
| **29** | 118 | 14 | noise (42%) | Plaid / schoolgirl skirt, bent-over or on-bed |
| **40** | 46 | 10 | noise (70%) | Heels, stockings, pantyhose |
| **30** | 27 | 5 | noise (44%) | Tight pants / leggings / jeans from behind, still clothed |
| **20** | 22 | 6 | noise (68%) | Sari, hijab, South Asian amateur interiors |

**24** and **29** are the cleanest proof that residual CLS works. Parent 27 already collected some lingerie; fishnets still group images the first cut had filed under other scene types and noise. Plaid skirt is not a parent type: it is a costume on top of several scene types. **20** is small but sharp, and one of the few residual groups that is directly usable for racial/cultural clothing.

### Body and identity

| Residual | n | n parents | Majority | Shared look |
|----------|--:|----------:|----------|-------------|
| **17** | 32 | 10 | noise (69%) | Pregnant belly, including illustrated (parent 0) |
| **48** | 144 | 12 | noise (46%) | Fuller / plus-size amateur body (not a breast-close-up leftover) |
| **35** | 103 | 12 | noise (42%) | Trans studio look (lingerie, studio branding) across parent sets |
| **39** | 27 | 7 | parent 18 (41%) | Same identity, more amateur / solo |
| **10** | 77 | 15 | noise (51%) | Tattoos as the shared surface; gay and hetero mixed |
| **16** | 42 | 9 | noise (38%) | Feet / soles (fetish crop, not parent-18 leftover) |

**17** is the best single example of a second CLS layer: pregnancy is a body attribute the parent cut never made a type of, and it still groups live-action with drawing.

### Setting

| Residual | n | n parents | Majority | Shared look |
|----------|--:|----------:|----------|-------------|
| **28** | 204 | 22 | noise (60%) | Outdoor / forest / public |
| **21** | 71 | 15 | noise (68%) | Kitchen / counter |
| **32** | 96 | 10 | noise (51%) | Gym / yoga / fitness; parent 14 is 31% |
| **18** | 129 | 11 | noise (76%) | Massage table / spa / clinical; gay and hetero |
| **25** | 23 | 6 | parent 4 (57%) | Bathroom / toilet / mirror |
| **41** | 22 | 6 | noise (45%) | Fairy lights / pink neon bedroom |

**28** is the headline setting cluster: 22 parent ids, and the montage is still woods, trail, snow, picnic. That is leftover appearance, not leftover images.

### Composition and practice

| Residual | n | n parents | Majority | Shared look |
|----------|--:|----------:|----------|-------------|
| **43** | 83 | 12 | noise (41%) | Cowgirl / squat, frontal, bright white interior |
| **52** | 84 | 12 | parent 14 (38%) | Bedroom from-behind; parent 15 is 23% |
| **62** | 57 | 8 | noise (32%) | Threesome on a bed (MMF), studio lighting |
| **66** | 22 | 7 | parent 17 (32%) | Threesome on a bed (MFF), studio lighting |
| **12** | 84 | 12 | noise (79%) | Rope / bondage / shibari |
| **13** | 108 | 14 | noise (55%) | Butt-plug close-up |
| **15** | 33 | 8 | noise (70%) | Squirting / liquid |
| **1** | 21 | 5 | noise (76%) | Leather / femdom / gothic interior |
| **23** | 31 | 8 | noise (61%) | Streamer overlay / gamer chair / VTuber UI |
| **19** | 26 | 6 | noise (81%) | Japanese amateur / JAV-home lighting |

**52** and **32** matter for interpretation: after subtracting “gay set” versus “hetero couple set,” the leftover is a shared pose or place (from-behind in a bedroom; yoga mat / gym). That is the intended `(parent, residual)` reading.

## Parent-0 splits (illustrated taxonomy, not residual mix)

Residuals **0–9** are almost entirely parent 0. They do not mix live-action parents. They split the illustrated island that the first cut treated as one blob. Keep them as a nested illustrated typology; do not cite them as evidence that residual clustering found a second live-action look.

| Residual | n | Majority parent 0 | Medium |
|----------|--:|------------------:|--------|
| **0** | 25 | 92% | Title cards, graphics, hypno, awards |
| **2** | 58 | 100% | 2D anime |
| **7** | 26 | 100% | Western cartoon / comic |
| **6** | 31 | 100% | AI / portrait illustration |
| **3** | 49 | 100% | 3D anime / CGI |
| **4** | 20 | 95% | Photoreal 3D |
| **5** | 21 | 95% | 3D, from-behind |
| **8** | 24 | 96% | Game-engine / Sims-like UI |
| **9** | 38 | 100% | 3D furry / creature |

## Skip

Coherent grids that are **parent leftovers**, not a second layer:

| Residual | Why |
|----------|-----|
| **53, 54, 56, 57, 63, 64, 37, 45** | Still parent 18 (POV oral / genital close-up). Residual 57 is 92% parent 18 |
| **50** | 86% parent 22 (vulva crop) |
| **44** | Oiled breasts, mostly parent 28 |
| **47, 65** | Rear / doggy, mostly parents 32 and 15 |
| **51, 42, 49** | Extreme genital crops from noise |

Mixed but **not one motif** (do not name as types): **36, 46, 34, 58, 59, 61, 67**. Amateur / couple piles, not a second look.

## How to use this in the study

A thumbnail in a kept residual group is a pair: **(parent scene type, residual look)**. Example: parent 15 (studio hetero couple) + residual 29 (plaid skirt) is a different claim from parent 32 (rear crop) + residual 29.

This is a second CLS layer, not a substitute for part-based (patch) labels. Claims about specific residual types should be grounded in the seed-42 montages of this leaf run, in the same way the parent typology is grounded in the working-cut montages.

### Six-grid figure

If only a few residual montages are shown, use these, in this order. Paths are under `data/dinov3_cls_clusters/nopca-n15-mcs20-ms20-residual-leaf/samples/`.

| Order | Residual | Grid | Why |
|------:|----------|------|-----|
| 1 | 24 | `cluster_24/_grid.jpg` | Fishnets across parent sets |
| 2 | 29 | `cluster_29/_grid.jpg` | Plaid skirt across parent sets |
| 3 | 28 | `cluster_28/_grid.jpg` | Outdoor setting; largest mixed residual |
| 4 | 17 | `cluster_17/_grid.jpg` | Pregnancy, live-action + drawing |
| 5 | 52 or 32 | `cluster_52/_grid.jpg` or `cluster_32/_grid.jpg` | Same composition across gay / hetero |
| 6 | 20 or 48 | `cluster_20/_grid.jpg` or `cluster_48/_grid.jpg` | Clothing (sari/hijab) or body (plus-size) |

Those six make the Stage 1 claim: subtracting the parent mean does not just re-slice the same sets; it recovers clothing, body, and setting that the 33-type cut never isolated.
