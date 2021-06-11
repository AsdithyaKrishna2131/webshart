import webshart
from huggingface_hub import get_token

dataset = webshart.discover_dataset(
    "NebulaeWis/e621-2024-webp-4Mpixel", hf_token=get_token()
)

# Quick stats (instant, uses cached values if available)
stats = dataset.get_stats()
