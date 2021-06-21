from webshart import MetadataExtractor
from huggingface_hub import get_token

# Create an extractor (optionally with HF token for private datasets)
extractor = MetadataExtractor(hf_token=get_token())

