import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ================== SETTINGS ==================
input_file = "data/sampled_data.csv"
output_file = "data/sampled_with_thumbnails.csv"
log_file = Path("data/scraper.log")
thumbnail_dir = Path("data/thumbnails")
delay = 5.0
CHECKPOINT_EVERY = 10
# =============================================

logger = logging.getLogger("scraper")


def setup_logging() -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def extract_viewkey(url: str) -> str | None:
    match = re.search(r"viewkey=([a-zA-Z0-9]+)", url)
    return match.group(1) if match else None


def thumbnail_path_for(viewkey: str) -> Path:
    return thumbnail_dir / f"{viewkey}.jpg"


def init_thumbnail_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "thumbnail_url" not in df.columns:
        df["thumbnail_url"] = None
    if "thumbnail_path" not in df.columns:
        df["thumbnail_path"] = None
    if "thumbnail_success" not in df.columns:
        df["thumbnail_success"] = False
    return df


def merge_missing_input_rows(df: pd.DataFrame, input_df: pd.DataFrame) -> pd.DataFrame:
    known_urls = set(df["url"].astype(str))
    missing = input_df[~input_df["url"].astype(str).isin(known_urls)]
    if missing.empty:
        return df

    missing = init_thumbnail_columns(missing)
    logger.info("Merged %d new rows from %s", len(missing), input_file)
    return pd.concat([df, missing], ignore_index=True)


def reconcile_disk_thumbnails(df: pd.DataFrame) -> int:
    """Mark rows successful when the thumbnail file already exists on disk."""
    reconciled = 0
    for idx, row in df.iterrows():
        if bool(df.at[idx, "thumbnail_success"]):
            continue

        viewkey = extract_viewkey(str(row["url"]))
        if not viewkey:
            continue

        path = thumbnail_path_for(viewkey)
        if path.exists():
            df.at[idx, "thumbnail_path"] = str(path)
            df.at[idx, "thumbnail_success"] = True
            reconciled += 1

    return reconciled


def load_working_frame() -> pd.DataFrame:
    input_df = pd.read_csv(input_file, on_bad_lines="skip")
    input_df = init_thumbnail_columns(input_df)

    if Path(output_file).exists():
        df = pd.read_csv(output_file, on_bad_lines="skip")
        df = init_thumbnail_columns(df)
        df = merge_missing_input_rows(df, input_df)
        logger.info(
            "Resuming from %s: %d/%d marked successful in CSV",
            output_file,
            int(df["thumbnail_success"].sum()),
            len(df),
        )
    else:
        df = input_df
        logger.info("Starting fresh from %s (%d rows)", input_file, len(df))

    reconciled = reconcile_disk_thumbnails(df)
    if reconciled:
        logger.info("Reconciled %d rows from existing files in %s", reconciled, thumbnail_dir)
        df.to_csv(output_file, index=False)

    on_disk = int(df["thumbnail_success"].sum())
    logger.info("Ready: %d/%d thumbnails available after resume checks", on_disk, len(df))
    return df


def checkpoint(df: pd.DataFrame, idx: int) -> None:
    if (idx + 1) % CHECKPOINT_EVERY == 0:
        df.to_csv(output_file, index=False)
        success = int(df["thumbnail_success"].sum())
        logger.info(
            "Checkpoint row %d/%d: %d successful (%.1f%%)",
            idx + 1,
            len(df),
            success,
            100 * success / len(df),
        )


def main() -> None:
    setup_logging()
    thumbnail_dir.mkdir(parents=True, exist_ok=True)

    run_started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info("=== Scraper run started %s (delay=%.1fs) ===", run_started, delay)

    df = load_working_frame()

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"})

    logger.info("Starting Hybrid Thumbnail Downloader (CDN → OG fallback)")

    attempted = 0
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading thumbnails"):
        if bool(df.at[idx, "thumbnail_success"]):
            continue

        url = str(row["url"])
        viewkey = extract_viewkey(url)
        if not viewkey:
            logger.warning("Row %d: no viewkey in URL, skipping", idx)
            continue

        thumbnail_path = thumbnail_path_for(viewkey)

        if thumbnail_path.exists():
            df.at[idx, "thumbnail_path"] = str(thumbnail_path)
            df.at[idx, "thumbnail_success"] = True
            checkpoint(df, idx)
            continue

        downloaded = False
        attempted += 1

        try:
            thumbnail_url = (
                f"https://ci.phncdn.com/videos/{viewkey[:3]}/{viewkey[3:6]}/"
                f"{viewkey[6:9]}/{viewkey}/thumbnail.jpg"
            )
            response = session.get(thumbnail_url, timeout=8)
            if response.status_code == 200 and "image" in response.headers.get("content-type", ""):
                thumbnail_path.write_bytes(response.content)
                df.at[idx, "thumbnail_url"] = thumbnail_url
                df.at[idx, "thumbnail_path"] = str(thumbnail_path)
                df.at[idx, "thumbnail_success"] = True
                downloaded = True
        except Exception as exc:
            logger.debug("CDN failed for %s: %s", viewkey, exc)

        if not downloaded:
            try:
                response = session.get(url, timeout=15)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    og_image = soup.find("meta", property="og:image")
                    if og_image and og_image.get("content"):
                        real_url = og_image["content"]
                        img_response = session.get(real_url, timeout=10)
                        if img_response.status_code == 200 and "image" in img_response.headers.get(
                            "content-type", ""
                        ):
                            thumbnail_path.write_bytes(img_response.content)
                            df.at[idx, "thumbnail_url"] = real_url
                            df.at[idx, "thumbnail_path"] = str(thumbnail_path)
                            df.at[idx, "thumbnail_success"] = True
                            downloaded = True
            except Exception as exc:
                logger.debug("OG fallback failed for %s: %s", viewkey, exc)

        if not downloaded:
            logger.info("Failed: %s (year=%s)", viewkey, row.get("year", "?"))

        checkpoint(df, idx)
        time.sleep(delay)

    df.to_csv(output_file, index=False)

    success = int(df["thumbnail_success"].sum())
    success_rate = 100 * success / len(df)
    remaining = len(df) - success
    logger.info("Done! Thumbnails saved to: %s", thumbnail_dir)
    logger.info(
        "Success rate: %.1f%% (%d/%d successful, %d remaining, %d download attempts this run)",
        success_rate,
        success,
        len(df),
        remaining,
        attempted,
    )
    logger.info("=== Scraper run finished %s ===", run_started)


if __name__ == "__main__":
    main()