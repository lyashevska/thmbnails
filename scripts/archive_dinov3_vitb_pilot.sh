#!/usr/bin/env bash
# Move ViT-B/16 pilot runs (1,743 valid thumbnails) out of active data/ dirs.
# Safe to re-run: skips paths that are already archived.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCHIVE="$ROOT/data/archive/dinov3_vitb16_pilot_1743"

move_if_exists() {
  local src="$1"
  local dest_dir="$2"
  if [[ -d "$src" ]]; then
    mkdir -p "$dest_dir"
    local name
    name="$(basename "$src")"
    if [[ -e "$dest_dir/$name" ]]; then
      echo "skip (already archived): $dest_dir/$name"
    else
      mv "$src" "$dest_dir/"
      echo "archived: $src -> $dest_dir/$name"
    fi
  fi
}

mkdir -p "$ARCHIVE"
cat > "$ARCHIVE/README.txt" <<'EOF'
ViT-B/16 pilot runs (facebook/dinov3-vitb16-pretrain-lvd1689m)
- 1,743 valid thumbnails from the 500/year sample
- Superseded by ViT-L/16 CLS runs on the full 12,500 / ~8,666 valid corpus

Contents:
  embeddings/       CLS run 20260617T091002Z
  clusters/         CLS cluster 20260617T123852Z
  patch_embeddings/ Patch run 20260618T125222Z
  patch_motifs/     Patch motifs 20260618T140719Z
EOF

move_if_exists "$ROOT/data/dinov3_embeddings/20260617T091002Z" "$ARCHIVE/embeddings"
move_if_exists "$ROOT/data/dinov3_clusters/20260617T123852Z" "$ARCHIVE/clusters"
move_if_exists "$ROOT/data/dinov3_patch_embeddings/20260618T125222Z" "$ARCHIVE/patch_embeddings"
move_if_exists "$ROOT/data/dinov3_patch_motifs/20260618T140719Z" "$ARCHIVE/patch_motifs"

echo "Done. Active data/dinov3_* dirs are ready for new ViT-L runs."