import re, os
from utils.misc import ensure_image_url, parser_answers_into_option
from vlmeval.smp import LMUDataRoot


def preprocess_api(message, model, processor, **kwargs):
    """
    You can implement your own preprocess function here like what we do in the `default_prepocess_function` of class `APIModelWrapper`  of models/base.py
    """
    return message


def postprocess_api(outputs, model, processor, **kwargs):
    assert (
        model is None and processor is None
    ), "Model and processor should be None in when using API mode."

    response = outputs

    if kwargs["dataset_name"] == "GUIElementGrounding":
        resp = response.split("assistantos\n")[-1]
    elif kwargs["dataset_name"] == "GUIContentUnderstanding":
        resp = response.split("assistantall\n")[-1]
    else:
        resp = response.split("assistant")[-1]
    lt = len(resp)
    counter, end = 1, None
    for i in range(lt):
        if resp[i] == "{":
            counter += 1
        elif resp[i] == "}":
            counter -= 1
        if counter == 0:
            end = i
            break
        elif i == lt - 1:
            end = lt
            break
    if end is not None:
        response = resp[:end]

    print(f"\033[32m{response}\033[0m")
    return response


def parse_gui_bbox_api(response, meta=None):
    match = re.search(r"x=([\d.]+), y=([\d.]+)", response)
    if match:
        click_point = [float(match.group(1)), float(match.group(2))]
    else:
        click_point = [0.0, 0.0]
    return click_point


def parse_gui_qa_api(response, meta=None):
    match = parser_answers_into_option(response)
    if match:
        return match
    else:
        return None


def build_prompt_example(line, dataset):
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
    if (
        dataset == "GUIElementGrounding"
    ):  # for example, we only build customized prompt for GUIElementGrounding
        msgs = []
        msgs.append(
            dict(
                role="system",
                type="text",
                value="You are a useful agent for computer use.",
            )
        )
        tgt_path = os.path.join(
            (
                os.environ["IMAGE_ROOT_DIR"]
                if os.path.exists(os.environ.get("IMAGE_ROOT_DIR", ""))
                else f"{LMUDataRoot()}/MMBench-GUI/offline_images"
            ),
            line["image_path"],
        )
        user_prompt = f"Please give me the position of the element that matchs the following description: {line['instruction']}"

        msgs.append(dict(role="user", type="image", value=tgt_path))
        msgs.append(dict(role="user", type="text", value=user_prompt))
        return msgs
