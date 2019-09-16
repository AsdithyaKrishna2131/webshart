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
                time.sleep(0.2)
            else:
                break

        try:
            yield next(loader)
        except StopIteration:
            break


class ShardCacheMonitor:
    """Helper for monitoring shard cache downloads with async support."""

    def __init__(self, dataloader):
        self.dataloader = dataloader

    async def wait_for_shard_async(self, filename, update_interval=0.1):
        """
        Async generator that yields download progress updates.

        Usage:
            async for progress in monitor.wait_for_shard_async('data-0000.tar'):
                print(f"Downloaded: {progress['downloaded']} / {progress['total']}")
        """
        import asyncio

        # Start the download
        self.dataloader.prepare_shard_by_name(filename)

        while True:
            status = self.dataloader.get_shard_cache_status(filename)
