import time
from tqdm import tqdm
from webshart import discover_dataset, TarDataLoader, CacheWaitContext


# Create dataloader with shard caching enabled
dataset = discover_dataset("NebulaeWis/e621-2024-webp-4Mpixel")
dataset.enable_shard_cache("cache/", cache_limit_gb=50.0)

