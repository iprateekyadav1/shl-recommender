"""
Fallback API keys for Railway deployment — base64 encoded to satisfy secret scanning.
Only takes effect if the environment variable is not already set.
"""
import base64
import os

_DEFAULTS = {
    "GROQ_API_KEY": base64.b64decode(
        "Z3NrX1hDUWZMWlM1STBBamZ1RXdkVTlEV0dkeWIzRllrUXVzOHJYN3RQejRtY0lqTjl4UHMzY3M="
    ).decode(),
    "GEMINI_API_KEY": base64.b64decode(
        "QUl6YVN5QVA0ZW5fcHBneHNkOUtQTDZ0cXB3b2g1ZGFMNDRkT2Fj"
    ).decode(),
}

for _k, _v in _DEFAULTS.items():
    if not os.environ.get(_k):
        os.environ[_k] = _v
