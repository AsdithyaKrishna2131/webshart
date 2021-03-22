from webshart import TarDataLoader, discover_dataset
from huggingface_hub import get_token
import os
import pickle

hf_token = get_token()
dataset = discover_dataset("NebulaeWis/e621-2024-webp-4Mpixel", hf_token=hf_token)
loader = TarDataLoader(dataset)

# Resume from checkpoint if it exists
checkpoint_file = "dataloader_state.pkl"
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "rb") as f:
        state = pickle.load(f)
