from webshart import TarDataLoader, discover_dataset
from huggingface_hub import get_token
import os
import pickle

hf_token = get_token()
dataset = discover_dataset("NebulaeWis/e621-2024-webp-4Mpixel", hf_token=hf_token)
loader = TarDataLoader(dataset)
