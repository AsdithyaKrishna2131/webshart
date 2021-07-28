"""
⚠️ This method uses several parallel connections to the source dataset, which will
result in **429 Too Many Requests** responses from providers like Hugging Face Hub.
It is intended for low-volume IO, with better solutions on the way for extended streaming.

Alternatively, if you host your own minIO or similar, feel free to use this API, it is fast.
"""

import webshart
from huggingface_hub import get_token

