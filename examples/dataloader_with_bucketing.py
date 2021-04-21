from webshart import BucketDataLoader, discover_dataset
from huggingface_hub import get_token
import os
import pickle

hf_token = get_token()
dataset = discover_dataset(
