import os
import re
import os.path as osp
from collections import defaultdict
from urllib.parse import urlparse, parse_qs, urlunparse


def load_env():
    import logging

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s - %(filename)s: %(funcName)s - %(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        import vlmeval
    except ImportError:
        logging.error(
            "VLMEval is not installed. Failed to import environment variables from .env file. "
        )
        return
    pth = osp.realpath(__file__)
    pth = osp.join(pth, "../../.env")
    pth = osp.realpath(pth)
    if not osp.exists(pth):
        logging.error(f"Did not detect the .env file at {pth}, failed to load. ")
        return

    from dotenv import dotenv_values

    values = dotenv_values(pth)
    for k, v in values.items():
        if v is not None and len(v):
            os.environ[k] = v
    logging.info(f"Extra Env variables successfully loaded from {pth}")


def deep_nested():
    return defaultdict(lambda: defaultdict(list))


def revert_defaultdict(obj):
    if isinstance(obj, defaultdict):
        return {k: revert_defaultdict(v) for k, v in obj.items()}
    elif isinstance(obj, dict):
        return {k: revert_defaultdict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [revert_defaultdict(i) for i in obj]
    else:
        return obj


def insert_at(d: dict, key: str, value, index: int) -> dict:
    items = list(d.items())
    items.insert(index, (key, value))
    return dict(items)


def ensure_image_url(image: str) -> str:
    prefixes = ["http://", "https://", "file://", "data:image;"]
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return "file://" + image
    raise ValueError(f"Invalid image: {image}")


def parser_answers_into_option(text, meta=None):
    patterns = [
        r"\b([A-F])[\.:](?!\w)",  # parser: A.  B:
        r"\bOption\s+([A-F])\b",  # Option A / Option B
        r"\bAnswer\s*[:：]?\s*([A-F])\b",  # Answer: A
        r"^[ \t]*([A-F])[\.:]?",  # the first letter in a row: A. or A:
        r"[\'\"]([A-F])[\'\"]",  # 'A' or "B"
        r"\b([A-F])\b(?!\s+\w)",  # a single alphabet A-F, without following word
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return None


def parser_response_into_coordinates(text, meta=None):
    """
    Default parse function for the response.
    It should be overridden by the user if needed.
    """
    pattern = r"""
        (?:x\s*[:=]?\s*)?                 
        [\(\[\{]?              
        \s*([-+]?(?:\d+\.\d+|\.\d+|\d+))\s* 
        [,\s;]+                       
        (?:y\s*[:=]?\s*)?        
        \s*([-+]?(?:\d+\.\d+|\.\d+|\d+))\s* 
        [\)\]\}]?    
    """

    matches = re.findall(pattern, text, re.IGNORECASE | re.VERBOSE)
    if len(matches) == 0:
        return None
    else:
        return [(float(x), float(y)) for x, y in matches][
            0
        ]  # we select the first if multiple (x,y) are parsed.


def get_api_info(model_path_or_class):
    assert (
        "http" in model_path_or_class
    ), "When using API, model_path_or_class should be a URL containing `http` or `https`."
    parsed = urlparse(model_path_or_class)

    query_params = parse_qs(parsed.query)
    api_key = query_params.get("api_key", ["api_key_placeholder"])[0]
    model_name = query_params.get("model", ["gpt-4o"])[0]

    clean_url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))

    return clean_url, api_key, model_name
