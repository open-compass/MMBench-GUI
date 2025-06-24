import ast
import re, os
from utils.misc import parser_answers_into_option
import math
from vlmeval.smp import LMUDataRoot

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


# ------------------------------------------- finish ----------------------------------------------------------


def build_custom_prompt(line, dataset):
    """
    Build prompts as you need.

    Args:
        line (dict), original data from dataloader.
                    An example for level1:
                    line={
                            "index":0,
                            "image_path": "os_ios/9e304d4e_5fdc3924_51c74094e7e217f384edd0d882ea6fb19b839ddc029893daa6dd17fafb49b3d6.png",
                            "question": "Based on the navigation elements, what can be inferred about the current screen's position in the app's hierarchy?",
                            "options": {
                                "A":"It's a sub-screen within a 'Rings' section",
                                "B":"It's the main dashboard of the app",
                                "C":"It's a sub-screen within the 'Summary' section",
                                "D":"It's a standalone 'Awards' page accessible from anywhere",
                                "E":"It's the 'Sharing' section of the app"
                            },
                            "answer": "C",
                            "explanation": "The green back arrow at the top left with 'Summary' indicates this is a sub-screen within the Summary section. The bottom navigation also shows 'Summary' highlighted, confirming we're in a sub-page (specifically 'Awards') within the Summary section, not on the main Summary page itself.",
                            "difficulty": "easy"
                            "image_size":[
                                1179,
                                2556
                            ],
                            "platform":"os_ios",
                            "app_name":"Fitness"
                    }

                    An example for level2:
                    line={
                            "index":0,
                            "image_path":"os_windows/0b08bd98_a0e7b2a5_68e346390d562be39f55c1aa7db4a5068d16842c0cb29bd1c6e3b49292a242d1.png",
                            "instruction":"The downward arrow button allows you to scroll down through the list of years.",
                            "bbox":[
                                0.3875,
                                0.1361,
                                0.3945,
                                0.1507
                            ],
                            "image_size":[
                                2560,
                                1440
                            ],
                            "data_type":"icon",
                            "platform":"os_windows",
                            "app_name":"calendar",
                            "grounding_type":"basic"
                    }
        dataset (str), the name of the benchmark. It can be used to determine different prompt format for different task.
                        It should be one of ["GUIElementGrounding", "GUIContentUnderstanding", "GUITaskAutomation", "GUITaskCollaboration": ,']
    Returns:
        msgs (list[dict]): inputs to model. It will be processed by preprocess_uitars provided by this file after some nessaccery checking.
    """
    msgs = []

    tgt_path = os.path.join(
        (
            os.environ["IMAGE_ROOT_DIR"]
            if os.path.exists(os.environ.get("IMAGE_ROOT_DIR", ""))
            else f"{LMUDataRoot()}/MMBench-GUI/offline_images"
        ),
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


def parse_grounding_response(response, meta):
    """Parse coordinates from model's response for evaluation

    Args:
        response (str), response from model. It is also the outputs of postprocess_uitars or our default postprocess function.
        meta (dict), original data from dataloader.

    Returns:
        parsed_predicted_point (list, None): The parsed coordinates of your prediction.
    """
    click_point = re.findall(r"\d+", response)
    if len(click_point) == 2:
        click_point = [int(x) for x in click_point]
        return uitars_postprocess(click_point, ast.literal_eval(meta["image_size"]))
    else:
        return None


def parse_understanding_response(response, meta):
    """Default parse function for the response. It should be overridden by the user if needed.

    Args:
        response (str), response from model. It is also the outputs of postprocess_uitars or our default postprocess function.
        meta (dict), original data from dataloader.

    Returns:
        match (str, None): The parsed option which is a single alphabet, for example, "B".
    """
    match = parser_answers_into_option(response)
    if match:
        return match
    else:
        return None
