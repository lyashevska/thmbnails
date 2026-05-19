import pandas as pd
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import time
from tqdm import tqdm
import re

# ================== SETTINGS ==================
input_file = "data/sampled_data.csv"
output_file = "data/sampled_with_thumbnails.csv"
thumbnail_dir = Path("data/thumbnails")
delay = 2.0                    # Increased for safety
CHECKPOINT_EVERY = 10          # flush output CSV after every N rows
# =============================================

thumbnail_dir.mkdir(exist_ok=True)

if Path(output_file).exists():
    df = pd.read_csv(output_file)
    already_done = int(df['thumbnail_success'].sum())
    print(f"Resuming: {already_done}/{len(df)} thumbnails already downloaded")
else:
    df = pd.read_csv(input_file)
    df['thumbnail_url'] = None
    df['thumbnail_path'] = None
    df['thumbnail_success'] = False

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)'})

print("Starting Hybrid Thumbnail Downloader (CDN → OG fallback)...\n")

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading thumbnails"):
    # Skip rows already successfully downloaded
    if df.at[idx, 'thumbnail_success'] == True:
        continue

    url = str(row['url'])

    viewkey_match = re.search(r'viewkey=([a-zA-Z0-9]+)', url)
    if not viewkey_match:
        continue
    viewkey = viewkey_match.group(1)

    thumbnail_path = thumbnail_dir / f"{viewkey}.jpg"

    # File exists on disk but CSV was not flushed on previous run
    if thumbnail_path.exists():
        df.at[idx, 'thumbnail_path'] = str(thumbnail_path)
        df.at[idx, 'thumbnail_success'] = True
        continue

    downloaded = False

    # Strategy 1: Try Direct CDN first (fast but often fails)
    try:
        thumbnail_url = f"https://ci.phncdn.com/videos/{viewkey[:3]}/{viewkey[3:6]}/{viewkey[6:9]}/{viewkey}/thumbnail.jpg"
        r = session.get(thumbnail_url, timeout=8)
        if r.status_code == 200 and 'image' in r.headers.get('content-type', ''):
            thumbnail_path.write_bytes(r.content)
            df.at[idx, 'thumbnail_url'] = thumbnail_url
            df.at[idx, 'thumbnail_path'] = str(thumbnail_path)
            df.at[idx, 'thumbnail_success'] = True
            downloaded = True
    except:
        pass

    # Strategy 2: Fallback to OG:image (more reliable)
    if not downloaded:
        try:
            r = session.get(url, timeout=15)
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                og_image = soup.find('meta', property='og:image')
                if og_image and og_image.get('content'):
                    real_url = og_image['content']
                    r_img = session.get(real_url, timeout=10)
                    if r_img.status_code == 200 and 'image' in r_img.headers.get('content-type', ''):
                        thumbnail_path.write_bytes(r_img.content)
                        df.at[idx, 'thumbnail_url'] = real_url
                        df.at[idx, 'thumbnail_path'] = str(thumbnail_path)
                        df.at[idx, 'thumbnail_success'] = True
                        downloaded = True
        except:
            pass

    if (idx + 1) % CHECKPOINT_EVERY == 0:
        df.to_csv(output_file, index=False)

    time.sleep(delay)

# Save results
df.to_csv(output_file, index=False)

success_rate = df['thumbnail_success'].mean() * 100
print(f"\nDone! Thumbnails saved to: {thumbnail_dir}")
print(f"Success rate: {success_rate:.1f}%")