# Methods

## Data sampling
To obtain a balanced temporal sample from the dataset, a custom script was developed to perform a stratified sample by year. The input dataset was loaded as a pandas DataFrame from a CSV file containing video metadata with a date column in YYYY-MM-DD format.

## Thumbnail Acquisition

Thumbnails corresponding to each video in the sampled dataset were retrieved using a **hybrid strategy** combining two complementary approaches.

### Direct CDN Construction (Strategy 1)
The first approach attempted to construct the thumbnail URL directly from the video’s unique `viewkey` extracted from the `url` column. Using known patterns of Pornhub’s Content Delivery Network (CDN), a direct image URL was generated (e.g., `https://ci.phncdn.com/videos/{viewkey[:3]}/{viewkey[3:6]}/{viewkey[6:9]}/{viewkey}/thumbnail.jpg`). The image was then downloaded via the `requests` library and saved locally in the `data/thumbnails/` directory as `{viewkey}.jpg`. This method was computationally lightweight and fast, requiring only a single HTTP request per video. However, it proved highly fragile due to frequent changes in Pornhub’s CDN structure, resulting in low reliability across runs.

### Open Graph Meta Tag Extraction (Strategy 2) 
Due to the instability of the direct CDN method, a second, more robust approach was ultimately adopted as the **primary method**. For each video, the full video page was fetched using the `requests` library. The representative thumbnail URL was then extracted from the `<meta property="og:image">` tag using the BeautifulSoup4 HTML parser. The identified image was subsequently downloaded and stored in the `data/thumbnails/` directory as `{viewkey}.jpg`.  

A fixed delay of 2.0 seconds was enforced between requests in both strategies to avoid excessive server load and respect server constraints.  

For the final dataset, three additional columns were appended:  
- `thumbnail_url` (remote image location)  
- `thumbnail_path` (local relative path to the downloaded file)  
- `thumbnail_success` (boolean indicating successful acquisition)  

This explicit linkage enables straightforward correspondence between textual video metadata and its visual thumbnail. The enriched dataset was exported as `sampled_with_thumbnails.csv`. A success rate of **48.2%** was achieved using the hybrid CDN → OG fallback approach.

**Note on Data Limitations**  
Some videos in the dataset were disabled or still processing at the time of scraping. These videos either contained **no thumbnail** or returned Pornhub’s generic placeholder image (“This video is still converting”). These issues will be addressed in a subsequent phase of data cleaning and validation.