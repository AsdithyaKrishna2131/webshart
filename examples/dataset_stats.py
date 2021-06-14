import webshart
from huggingface_hub import get_token

dataset = webshart.discover_dataset(
    "NebulaeWis/e621-2024-webp-4Mpixel", hf_token=get_token()
)

# Quick stats (instant, uses cached values if available)
stats = dataset.get_stats()
print(f"Total shards: {stats['total_shards']}")
print(f"Estimated total files: {stats.get('total_files', 'Unknown')}")

# Detailed stats (loads all metadata)
