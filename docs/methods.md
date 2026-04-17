# Methods

## Data sampling
To obtain a balanced temporal sample from the dataset, a custom script was developed to perform a stratified sample by year. The input dataset was loaded as a pandas DataFrame from a CSV file containing video metadata with a date column in YYYY-MM-DD format.

## Thumbnail Acquisition
Thumbnails corresponding to each video in the sampled dataset were retrieved using two different strategies.

### Direct CDN Construction (Strategy 1)
The first approach attempted to construct the thumbnail URL directly from the video’s unique viewkey extracted from the url column. Using known patterns of Pornhub’s content delivery network (CDN), a direct image URL was generated. The image was then downloaded via the requests library and saved locally. This method was computationally lightweight and fast, requiring only a single HTTP request per video. However, it proved highly fragile; after changes in Pornhub’s CDN structure, the success rate dropped. 

### Open Graph Meta Tag Extraction (Strategy 2)
Due to the instability of the direct CDN method, a second, more robust approach was adopted. For each video, the full video page was fetched using the requests library. The thumbnail URL was then extracted from the <meta property="og:image"> tag using the BeautifulSoup4 library. The identified image was subsequently downloaded and stored in a year-organized directory structure (thumbnails/YYYY/{viewkey}.jpg).
A fixed delay of 2.0 seconds was enforced between requests in both strategies to avoid excessive server load.
For the final dataset, two additional columns were appended: thumbnail_url (remote image location) and thumbnail_path (local relative path to the downloaded file). This explicit linkage enables straightforward correspondence between textual video metadata and its visual thumbnail. The enriched dataset was exported as sampled_with_thumbnails.csv.