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
delay = 1.0          # seconds between requests
# =============================================

thumbnail_dir.mkdir(exist_ok=True)

df = pd.read_csv(input_file)

df['thumbnail_url'] = None
df['thumbnail_path'] = None
df['thumbnail_success'] = False

session = requests.Session()
session.headers.update({'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)'})

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading thumbnails"):
    url = row['url']
    viewkey_match = re.search(r'viewkey=([a-zA-Z0-9]+)', url)
    if not viewkey_match:
        continue
    viewkey = viewkey_match.group(1)
    
    # Try direct thumbnail (common pattern)
    thumbnail_url = f"https://ci.phncdn.com/videos/{viewkey[:3]}/{viewkey[3:6]}/{viewkey[6:9]}/{viewkey}/thumbnail.jpg"
    
    try:
        # First attempt: direct download
        r = session.get(thumbnail_url, timeout=10)
        if r.status_code == 200 and 'image' in r.headers.get('content-type', ''):
            thumbnail_path = thumbnail_dir / f"{viewkey}.jpg"
            thumbnail_path.write_bytes(r.content)
            df.at[idx, 'thumbnail_url'] = thumbnail_url
            df.at[idx, 'thumbnail_path'] = str(thumbnail_path)
            df.at[idx, 'thumbnail_success'] = True
            continue
    except:
        pass
    
    # Fallback: scrape og:image from page
    try:
        r = session.get(url, timeout=15)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, 'html.parser')
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                real_url = og_image['content']
                r_img = session.get(real_url, timeout=10)
                if r_img.status_code == 200:
                    thumbnail_path = thumbnail_dir / f"{viewkey}.jpg"
                    thumbnail_path.write_bytes(r_img.content)
                    df.at[idx, 'thumbnail_url'] = real_url
                    df.at[idx, 'thumbnail_path'] = str(thumbnail_path)
                    df.at[idx, 'thumbnail_success'] = True
    except:
        pass
    
    time.sleep(delay)   # Be polite to the server

# Save updated dataset
df.to_csv(output_file, index=False)
print(f"\nDone! Thumbnails saved to: {thumbnail_dir}")
print(f"Success rate: {df['thumbnail_success'].mean():.1%}")