import ast
import torch
import re, os
from utils.misc import ensure_image_url, parser_answers_into_option
import math

# ------------------------ constants, copied from official repo of uitars --------------------------
IMAGE_FACTOR = 28
MIN_PIXELS = 100 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768

COMPUTER_USE_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
```
Thought: ...
Action: ...
```

## Action Space

click(point='<point>x1 y1</point>')
left_double(point='<point>x1 y1</point>')
right_single(point='<point>x1 y1</point>')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
hotkey(key='ctrl c') # Split keys with a space and use lowercase. Also, do not use more than 3 keys in one hotkey action.
type(content='xxx') # Use escape characters \\', \\\", and \\n in content part to ensure we can parse the content in normal python string format. If you want to submit your input, use \\n at the end of content. 
scroll(point='<point>x1 y1</point>', direction='down or up or right or left') # Show more information on the `direction` side.
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

MOBILE_USE_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 
## Output Format
```
Thought: ...
Action: ...
```
## Action Space

click(point='<point>x1 y1</point>')
long_press(point='<point>x1 y1</point>')
type(content='') #If you want to submit your input, use "\\n" at the end of `content`.
scroll(point='<point>x1 y1</point>', direction='down or up or right or left')
open_app(app_name=\'\')
drag(start_point='<point>x1 y1</point>', end_point='<point>x2 y2</point>')
press_home()
press_back()
finished(content='xxx') # Use escape characters \\', \\", and \\n in content part to ensure we can parse the content in normal python string format.


## Note
- Use {language} in `Thought` part.
- Write a small plan and finally summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{instruction}
"""

GROUNDING_DOUBAO = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction
{instruction}"""


# ------------------------ utils, copied from official repo of uitars --------------------------
def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def uitars_postprocess(pred_point, size):
    model_output_width = pred_point[0]
    model_output_height = pred_point[1]

    width, height = size

    new_height, new_width = smart_resize(height, width)
    new_coordinate = (model_output_width / new_width, model_output_height / new_height)

    return new_coordinate


# ---------------------------------------------------------------------------------------------------


def preprocess_uitars(message, model, processor, **kwargs):
    from qwen_vl_utils import process_vision_info

    messages = []
    if "system" == message[0]["role"]:
        messages.append({"role": "system", "content": message[0]["value"]})
        message = message[1:]
    messages.append(
        {"role": "user", "content": prepare_content(message, processor, **kwargs)}
    )

    text = processor.apply_chat_template(
        [messages], tokenize=False, add_generation_prompt=True
    )

    images, videos = process_vision_info([messages])
    inputs = processor(
        text=text, images=images, videos=videos, padding=True, return_tensors="pt"
    )
    inputs = inputs.to("cuda")

    return inputs


def postprocess_uitars(outputs, model, processor, **kwargs):
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().numpy()
    out = processor.batch_decode(
        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    response = out[0]
    resp = response.split("assistant\n")[-1]
    print(f"\033[32m{resp}\033[0m")
    return resp


def build_custom_prompt(line, dataset):
    msgs = []

    tgt_path = os.path.join(
        "/mnt/petrelfs/wangxuehui/project/computer_use/MMBench-GUI/release/offline_images",
        line["image_path"],
    )
    instruction = line["instruction"]

    if dataset == "GUIElementGrounding":
        msgs.append({"role": "user", "type": "image", "value": f"{tgt_path}"})

        msgs.append(
            {
                "role": "user",
                "type": "text",
                "value": GROUNDING_DOUBAO.format(instruction=instruction),
            }
        )
        return msgs


def prepare_content(inputs, processor, **kwargs):
    """
    inputs list[dict[str, str]], each dict has keys: ['type', 'value']
    """

    content = []
    for s in inputs:
        if s["type"] == "image":
            item = {"type": "image", "image": ensure_image_url(s["value"])}

            item["min_pixels"] = MIN_PIXELS
            item["max_pixels"] = MAX_PIXELS
        elif s["type"] == "text":
            item = {"type": "text", "text": s["value"]}
        elif s["type"] == "audio":
            item = {"type": "audio", "audio": s["value"]}
        else:
            raise ValueError(f"Invalid message type: {s['type']}, {s}")
        content.append(item)
    return content


def parse_grounding_response(response, meta):

    click_point = re.findall(r"\d+", response)
    if len(click_point) == 2:
        click_point = [int(x) for x in click_point]
        return uitars_postprocess(click_point, ast.literal_eval(meta["image_size"]))
    else:
        return None


def parse_understanding_response(response, meta):
    """
    Default parse function for the response.
    It should be overridden by the user if needed.
    """
    match = parser_answers_into_option(response)
    if match:
        return match
    else:
        return None
