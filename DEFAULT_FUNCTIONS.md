# Default functions and format

We provide default functions and format in our benchmark, including prompt building, preprocessing, postprocessing and parsing functions. We elaborate these functions and their outputs format here, and thus if it satisfy your requirement, you only need to set a system prompt, user prompt template and parsing functions. Our parsing functions are based on regex and it is robust I think, so maybe you even do not need to implement parsing function.  

## Prompt construction

### L1 - GUI Content Understanding

function:

```python
L1_SYSTEM_PROMPT_DEFAULT="""You are a GUI agent. You are given a screenshot of an application, a question and corresponding options. You need to choose one option as your answer for the question. Finally, you are ONLY allowed to return the single letter of your choice."""
def build_prompt(self, line, use_system=True, custom_system_prompt=None):

    tgt_path = self.dump_image(line)
    question = line["question"]
    options = {cand: line["options"][cand] for cand in string.ascii_uppercase}

    options_prompt = "Options:\n"
    for key, item in options.items():
        options_prompt += f"{key}. {item}\n"

    user_prompt = ""
    user_prompt += f"Question: {question}\n"
    if len(options):
        user_prompt += options_prompt
        user_prompt += "Please select the correct answer from the options above. \n"

    msgs = []
    if use_system:
        system_prompt = (
            L1_SYSTEM_PROMPT_DEFAULT
            if (custom_system_prompt is None) or (custom_system_prompt == "")
            else custom_system_prompt
        )
        msgs.append(dict(role="system", type="text", value=system_prompt))
    msgs = [dict(role="user", type="image", value=tgt_path)]
    msgs.append(dict(role="user", type="text", value=user_prompt))

    return msgs
```

output format (we enable system prompt for example):
```json
[
    {
        "role": "system",
        "type": "text",
        "value": "You are a GUI agent. You are given a screenshot of an application, a question and corresponding options. You need to choose one option as your answer for the question. Finally, you are ONLY allowed to return the single letter of your choice."
    },
    {
        "role": "user",
        "type": "image",
        "value": "/path/of/your/dataroot/os_ios/9e304d4e_5fdc3924_51c74094e7e217f384edd0d882ea6fb19b839ddc029893daa6dd17fafb49b3d6.png"
    },
    {
        "role": "user",
        "type": "text",
        "value": """Question: Based on the navigation elements, what can be inferred about the current screen's position in the app's hierarchy?
                    Options:
                    A. It's a sub-screen within a 'Rings' section
                    B. It's the main dashboard of the app
                    C. It's a sub-screen within the 'Summary' section
                    D. It's a standalone 'Awards' page accessible from anywhere
                    E. It's the 'Sharing' section of the app
                    Please select the correct answer from the options above."""
    },
]
```

### L2 - GUI Element Grounding

function:

```python
L2_SYSTEM_PROMPT_DEFAULT = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to finish this task following instructions from users."""
L2_USER_PROMPT_DEFAULT = """Output only the coordinate (x,y) of one point in your response. What element matches the following task: {instruction}"""

def build_prompt(self, line, use_system=True, custom_system_prompt=None):
    tgt_path = self.dump_image(line)

    user_prompt_template = os.environ.get("L2_USER_PROMPT", L2_USER_PROMPT_DEFAULT)
    user_prompt = user_prompt_template.format(instruction=line["instruction"])
    
    msgs = []
    if use_system:
        system_prompt = (
            L2_SYSTEM_PROMPT_DEFAULT
            if (custom_system_prompt is None) or (custom_system_prompt == "")
            else custom_system_prompt
        )
        msgs.append(dict(role="system", type="text", value=system_prompt))

    msgs = [dict(role="user", type="image", value=tgt_path)]
    msgs.append(dict(role="user", type="text", value=user_prompt))

    return msgs
```

output format (we enable system prompt for example):
```json
[
    {
        "role": "system",
        "type": "text",
        "value": "You are a GUI agent. You are given a task and a screenshot of the screen. You need to finish this task following instructions from users."
    },
    {
        "role": "user",
        "type": "image",
        "value": "/path/of/your/dataroot/os_windows/0b08bd98_a0e7b2a5_68e346390d562be39f55c1aa7db4a5068d16842c0cb29bd1c6e3b49292a242d1.png"
    },
    {
        "role": "user",
        "type": "text",
        "value": "Output only the coordinate (x,y) of one point in your response. What element matches the following task: The downward arrow button allows you to scroll down through the list of years."
    },
]
```

## Preprocess

### Local deployment model

function:
```python
def default_preprocess_function(self, message, model, processor, **kwargs):
    # We apply `process_vision_info` function of qwen_vl_utils as our default process function for visual part.
    from qwen_vl_utils import process_vision_info

    messages = []
    if "system" == message[0]["role"]:
        messages.append({"role": "system", "content": message[0]["value"]})
        message = message[1:]
    messages.append(
        {
            "role": "user",
            "content": simple_prepare_content(message, processor, **kwargs),
        }
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

def simple_prepare_content(inputs, processor, **kwargs):
    content = []
    for s in inputs:
        if s["type"] == "image":
            item = {"type": "image", "image": ensure_image_url(s["value"])}
        elif s["type"] == "text":
            item = {"type": "text", "text": s["value"]}
        elif s["type"] == "audio":
            item = {"type": "audio", "audio": s["value"]}
        else:
            raise ValueError(f"Invalid message type: {s['type']}, {s}")
        content.append(item)
    return content
```

### API-based model

output format (we enable system prompt for example):
```json
[
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a GUI agent. You are given a task and a screenshot of the screen. You need to finish this task following instructions from users."
            }
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": "data:image/png;base64,xxxxxxxxxxxxxxxxxxxxxxxxx......xxxxxxxxxxxxxxxxxx"
            },
            {
                "type": "text",
                "text": "Output only the coordinate (x,y) of one point in your response. What element matches the following task: The downward arrow button allows you to scroll down through the list of years."
            }
        ]
    }
]
```

## Postprocess

### Local deployment model

We decode the predicted ids into readable text strings and then directly return them.

```python
def default_postprocess_function(self, message, model, processor, **kwargs):
    outputs = outputs.cpu().numpy()
    out = processor.batch_decode(outputs, skip_special_tokens=True)

    response = out[0]
    return response
```


### API-based model

We directly return the response from api. 

```python
def default_postprocess_function(self, message, model, processor, **kwargs):
    # Due to various response format when calling api model, we directly return the response as our default implementation.
    # Thus, the desired output used for calculating matrics should be processed by `parse_function_xxxxxx` functions implemented in your model file.
    return message
```

## Parsing functions

### L1 - GUI Content Understanding

```python
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
```

### L2 - GUI Element Grounding

```python
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
```