# Methods

## Data sampling
To obtain a balanced temporal sample from the dataset, a custom script was developed to perform a stratified sample by year. The input dataset was loaded as a pandas DataFrame from a CSV file containing video metadata with a date column in YYYY-MM-DD format.

## Thumbnail Acquisition
Thumbnails were downloaded and organized by publication year. For each video in the sampled dataset, the unique viewkey was extracted from the URL. A direct thumbnail image URL was constructed using Pornhub’s content delivery network. The image was retrieved using the requests library and saved in a year-specific subdirectory (thumbnails/YYYY/) as {viewkey}.jpg.
Two columns were appended to the dataset: thumbnail_url (remote location) and thumbnail_path (local relative path). This explicit linkage enables straightforward correspondence between video metadata and its visual thumbnail. Requests were spaced with a 1.5-second delay to avoid server overload. The final enriched dataset was exported as sampled_with_thumbnails.csv.