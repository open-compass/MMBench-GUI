import os
import re
from collections import defaultdict
from urllib.parse import urlparse, parse_qs, urlunparse



def deep_nested():
    return defaultdict(lambda: defaultdict(list))

def insert_at(d: dict, key: str, value, index: int) -> dict:
    items = list(d.items())
    items.insert(index, (key, value))
    return dict(items)

def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')

def parser_answers_into_option(text):
    patterns = [
        r'\b([A-F])[\.:](?!\w)',                    # parser: A.  B:
        r'\bOption\s+([A-F])\b',                    # Option A / Option B
        r'\bAnswer\s*[:：]?\s*([A-F])\b',            # Answer: A 
        r'^[ \t]*([A-F])[\.:]?',                    # the first letter in a row: A. or A:
        r'[\'\"]([A-F])[\'\"]',                     # 'A' or "B"
        r'\b([A-F])\b(?!\s+\w)',                    # a single alphabet A-F, without following word
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return None

def get_api_info(model_path_or_class):
    assert 'http' in model_path_or_class, "When using API, model_path_or_class should be a URL containing `http` or `https`."
    parsed = urlparse(model_path_or_class)

    query_params = parse_qs(parsed.query)
    api_key = query_params.get('api_key', ['api_key_placeholder'])[0]
    model_name = query_params.get('model', ['gpt-4o'])[0]


    clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))

    return clean_url, api_key, model_name