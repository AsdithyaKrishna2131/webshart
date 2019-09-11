import time


def next_with_cache_wait(loader):
    """Drop-in replacement for next() that waits for cache if needed."""
    import time

    # Wait for cache if needed
    while loader.will_block():
        shard_info = loader.get_next_shard_info()
        if shard_info:
            loader.prepare_next_shard()
            # Simple progress indication
            print(f"Waiting for {shard_info['name']}...", end="", flush=True)
            while loader.will_block():
                time.sleep(0.2)
                print(".", end="", flush=True)
            print(" ready!")

    return next(loader)


def iter_with_cache_wait(loader, start_idx, end_idx):
    """Iterator that waits for cache when needed."""
    import time

    for idx in range(start_idx, end_idx + 1):
        # Wait for cache if needed
        while loader.will_block():
            shard_info = loader.get_next_shard_info()
            if (
                shard_info
                and not loader.get_shard_cache_status(shard_info["name"])["is_cached"]
            ):
                loader.prepare_next_shard()
