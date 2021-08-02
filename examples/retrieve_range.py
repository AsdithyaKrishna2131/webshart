"""
⚠️ This method uses several parallel connections to the source dataset, which will
result in **429 Too Many Requests** responses from providers like Hugging Face Hub.
It is intended for low-volume IO, with better solutions on the way for extended streaming.

Alternatively, if you host your own minIO or similar, feel free to use this API, it is fast.
"""

import webshart
from huggingface_hub import get_token

dataset = webshart.discover_dataset(
    "NebulaeWis/e621-2024-webp-4Mpixel", hf_token=get_token()
)

# Read files 0-100 from each of the first 10 shards
requests = []
for shard_idx in range(10):
    for file_idx in range(100):
