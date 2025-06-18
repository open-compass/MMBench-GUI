import torch
import re
from utils.misc import ensure_image_url, parser_answers_into_option

CHAT_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}{% endif %}"

def preprocess_aguvis(message, model, processor, **kwargs):
    from qwen_vl_utils import process_vision_info
    messages = []
    if 'system' == message[0]['role']:
        messages.append({'role': 'system', 'content': message[0]['value']})
        message = message[1:]
    messages.append({'role': 'user', 'content': prepare_content(message, processor, **kwargs)})
    text = processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True, chat_template=CHAT_TEMPLATE)
    if isinstance(text, list):
        text = text[0]

    if kwargs['dataset_name'] == 'GUIElementGrounding':
        text=[text + "<|im_start|>assistant<|recipient|>os\n"]
    elif kwargs['dataset_name'] == 'GUIContentUnderstanding':
        text=[text + "<|im_start|>assistant<|recipient|>all\n"]
    else:
        text = [text + "<|im_start|>assistant<|recipient|>"]
    images, videos = process_vision_info([messages])
    inputs = processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
    inputs = inputs.to('cuda')
    return inputs

def postprocess_aguvis(outputs, model, processor, **kwargs):
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().numpy()
    out = processor.batch_decode(outputs, skip_special_tokens=True)

    response = out[0]

    if kwargs['dataset_name'] == 'GUIElementGrounding':
        resp = response.split('assistantos\n')[-1]
    elif kwargs['dataset_name'] == 'GUIContentUnderstanding':
        resp = response.split('assistantall\n')[-1]
    else:
        resp = response.split('assistant')[-1]
    lt = len(resp)
    counter, end = 1, None
    for i in range(lt):
        if resp[i] == '{':
            counter += 1
        elif resp[i] == '}':
            counter -= 1
        if counter == 0:
            end = i
            break
        elif i == lt - 1:
            end = lt
            break
    if end is not None:
        response = resp[:end]


    print(f'\033[32m{response}\033[0m')
    return response


def prepare_content(inputs, processor, **kwargs):
    """
    inputs list[dict[str, str]], each dict has keys: ['type', 'value']
    """

    min_pixels = kwargs['min_pixels'] if 'min_pixels' in kwargs else processor.image_processor.min_pixels
    max_pixels = kwargs['max_pixels'] if 'max_pixels' in kwargs else processor.image_processor.max_pixels

    content = []
    for s in inputs:
        if s['type'] == 'image':
            item = {'type': 'image', 'image': ensure_image_url(s['value'])}

            item['min_pixels'] = min_pixels
            item['max_pixels'] = max_pixels
        elif s['type'] == 'text':
            item = {'type': 'text', 'text': s['value']}
        elif s['type'] == 'audio':
            item = {'type':'audio','audio':s['value']}
        else:
            raise ValueError(f"Invalid message type: {s['type']}, {s}")
        content.append(item)
    return content

def parse_gui_bbox_aguvis(response):
    match = re.search(r"x=([\d.]+), y=([\d.]+)", response)
    if match:
        click_point = [float(match.group(1)), float(match.group(2))]
    else:
        click_point = [0.0, 0.0]
    return click_point

def parse_gui_qa_aguvis(response):
    match = parser_answers_into_option(response)
    if match:
        return match
    else:
        return None
    

def build_prompt_example(line):
    """
    This is just a example function to build a prompt for the model.
    It should be replaced with the actual implementation.
    Args:
        line (str): The line to build the prompt from.
    Returns:
        msgs: list[dict]: A list of dictionaries representing the messages.
                It follows the format:
                [
                    {'role': 'xxxxx', 'type': 'image/text', content: 'xxxxx},
                    {'role': 'xxxxx', 'type': 'image/text', content: 'xxxxx},
                    ...
                    {'role': 'xxxxx', 'type': 'image/text', content: 'xxxxx}
                ]
    """
    pass