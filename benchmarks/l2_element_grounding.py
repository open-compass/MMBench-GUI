from collections import defaultdict
import os
import re
import pandas as pd
import json
import ast
import numpy as np
from typing import Union
from tqdm import tqdm
from PIL import Image

from vlmeval.dataset import ImageBaseDataset
from vlmeval.smp import LMUDataRoot, get_logger, load, read_ok, toliststr
from vlmeval.smp import dump
from ipdb import set_trace as st
from utils.import_utils import dynamic_import_function
from utils.misc import deep_nested, parser_response_into_coordinates
from .matrics import level2_calculate_scores

logger = get_logger("RUN")

L2_SYSTEM_PROMPT_DEFAULT = """You are a GUI agent. You are given a task and a screenshot of the screen. You need to finish this task following instructions from users."""
L2_USER_PROMPT_DEFAULT = """Output only the coordinate (x,y) of one point in your response. What element matches the following task: {instruction}"""


class GUIElementGrounding(ImageBaseDataset):
    MODALITY = "IMAGE"
    TYPE = "GUI_Element_Grounding"
    DATASET_URL = {
        "MMBench-GUI_L2": "xxxxxx.tsv",  # noqa
    }  # path
    DATASET_MD5 = {
        "MMBench-GUI_L2": "a2e042302ac389299873a857b7370a35",
    }
    RE_TYPE = "functional"

    def __init__(
        self,
        mode: str = "all",
        skip_noimg: bool = True,
        parse_function: str = None,
        **kwargs,
    ):
        assert mode in [
            "all",
            "basic",
            "advanced",
        ], "mode must be one of ['all', 'basic', 'advanced']"
        self.dataset_name = "GUIElementGrounding"
        self.mode = mode
        self.skip_noimg = skip_noimg
        self.parse_function = (
            dynamic_import_function(parse_function)
            if parse_function is not None
            else self.default_parse_response
        )
        self.kwargs = kwargs

        ROOT = LMUDataRoot()

        self.img_root = os.path.join(ROOT, "MMBench-GUI", "offline_images")

        data = self.load_data()
        if mode != "all":
            data = data[data["grounding_type"] == mode]
        if "image_path" in data:
            data["image_path"] = [
                f"{platform}/{image_path}"
                for platform, image_path in zip(data["platform"], data["image_path"])
            ]

        if self.skip_noimg:
            data = data[~pd.isna(data["image_path"])]

        self.data = data

    def default_parse_response(self, response):
        match = parser_response_into_coordinates(response)
        if match:
            assert len(match) == 2 and isinstance(match, tuple)
            return match
        else:
            return None

    def load_data(self):
        if not os.path.exists(
            os.path.join(LMUDataRoot(), "MMBench-GUI", "L2_annotations.json")
        ):
            url = self.DATASET_URL.get("MMBench-GUI_L2", None)
            file_md5 = (
                self.DATASET_MD5["MMBench-GUI_L2"]
                if "MMBench-GUI_L2" in self.DATASET_MD5
                else None
            )
        else:
            url = os.path.join(LMUDataRoot(), "MMBench-GUI", "L2_annotations.json")
            file_md5 = None

        return self.prepare_tsv(url, file_md5)

    def prepare_tsv(self, url, file_md5=None):
        # st()
        if "http" in url:
            data = super().prepare_tsv(url=url, file_md5=file_md5)
            if not isinstance(data, pd.DataFrame) and isinstance(
                data, Union[dict, list]
            ):
                data = pd.DataFrame(data)
            return data
        else:
            return pd.DataFrame(json.load(open(url, "r", encoding="utf-8")))

    def dump_image(self, line):
        assert "image_path" in line
        tgt_path = toliststr(line["image_path"])
        read_ok_flag = [read_ok(x) for x in tgt_path]
        # Might be the Relative Path
        if not all(read_ok_flag):
            tgt_path_abs = [os.path.join(self.img_root, x) for x in tgt_path]
            read_ok_flag = [read_ok(x) for x in tgt_path_abs]
            assert (
                read_ok_flag
            ), f"Field `image` is missing and we could not find {tgt_path} both as absolute or relative paths. "  # noqa
            tgt_path = tgt_path_abs

        return tgt_path

    def build_prompt(self, line, use_system=True, custom_system_prompt=None):
        # st()
        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)

        # we first check if `L2_USER_PROMPT` is set in the environment variables, if not, we use the default user prompt
        user_prompt_template = os.environ.get("L2_USER_PROMPT", L2_USER_PROMPT_DEFAULT)
        user_prompt = user_prompt_template.format(instruction=line["instruction"])
        msgs = []
        # add system prompt
        if use_system:
            system_prompt = (
                L2_SYSTEM_PROMPT_DEFAULT
                if (custom_system_prompt is None) or (custom_system_prompt == "")
                else custom_system_prompt
            )
            msgs.append(dict(role="system", type="text", value=system_prompt))
        if isinstance(tgt_path, list):
            msgs.extend([dict(role="user", type="image", value=p) for p in tgt_path])
        else:
            msgs = [dict(role="user", type="image", value=tgt_path)]
        msgs.append(dict(role="user", type="text", value=user_prompt))
        return msgs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(self.data.iloc[idx])

    def evaluate(self, eval_file, **judge_kwargs):
        stats = defaultdict(deep_nested)
        result = []

        data = load(eval_file)
        assert "bbox" in data and "prediction" in data
        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        for i in tqdm(range(len(lines))):
            line = lines[i]
            bbox = (
                line["bbox"]
                if isinstance(line["bbox"], list)
                else ast.literal_eval(line["bbox"])
            )
            # The format of bbox is (x1, y1, w2, y2)
            # x1, y1, w2, y2 = bbox
            # bbox = (x1, y1, x1 + w - 1, y1 + h - 1)

            image = Image.open(os.path.join(self.img_root, line["image_path"]))
            img_size = image.size

            def make_safe(value):
                if value == -1:
                    # we can tolerate -1 as a special value and nomalize it to 0
                    return 0
                else:
                    return value

            def is_normalized(value):
                if isinstance(value, float) or isinstance(value, int):
                    return 0 <= value <= 1
                elif isinstance(value, list) or isinstance(value, tuple):
                    return all(0 <= x <= 1 for x in value)

            if is_normalized(bbox):
                # If the bbox is already normalized, we do not need to normalize it again
                bbox = [
                    make_safe(bbox[0]),
                    make_safe(bbox[1]),
                    make_safe(bbox[2]),
                    make_safe(bbox[3]),
                ]
            else:
                bbox = [
                    make_safe(bbox[0]) / img_size[0],
                    make_safe(bbox[1]) / img_size[1],
                    make_safe(bbox[2]) / img_size[0],
                    make_safe(bbox[3]) / img_size[1],
                ]

            if any([x < 0 or x > 1 for x in bbox]):
                raise ValueError(
                    f"bbox out of range: {bbox} | {line['bbox']} | {img_size}"
                )

            key = (
                line["data_type"]
                if "category" not in line
                else line["category"] + ":" + line["data_type"]
            )
            prediction = str(line["prediction"])
            try:
                click_point = self.parse_function(prediction, meta=line)
                # Do Normalization By Default
                if click_point[0] > 1 or click_point[1] > 1:
                    click_point = (
                        click_point[0] / img_size[0],
                        click_point[1] / img_size[1],
                    )

                match = (bbox[0] <= click_point[0] <= bbox[2]) and (
                    bbox[1] <= click_point[1] <= bbox[3]
                )

                if match:
                    # stats[line["platform"]][line["grounding_type"]][key].append(1)
                    stats[line["grounding_type"]][line["platform"]][key].append(1)

                else:
                    # stats[line["platform"]][line["grounding_type"]][key].append(0)
                    stats[line["grounding_type"]][line["platform"]][key].append(0)

                is_wrong_format = False

            except Exception as e:
                logger.warning(f"exception in screenspot eval:{e}")
                # stats[line["platform"]][line["grounding_type"]][key].append(-1)
                stats[line["grounding_type"]][line["platform"]][key].append(-1)

                match, is_wrong_format, click_point = False, True, None

            result.append(
                {
                    "img_path": os.path.join(self.img_root, line["image_path"]),
                    "instruction": line["instruction"],
                    "bbox": line["bbox"],
                    "parsed_bbox(format:x1,y1,x2,y2)": bbox,
                    "type": line["data_type"],
                    "app_name": line["app_name"],
                    "platform": line["platform"],
                    "match": 1 if match else 0,
                    "is_wrong_format": is_wrong_format,
                    "pred_bbox": click_point,
                    "grounding_type": line["grounding_type"],
                }
            )
        # st()
        final_score_dict = level2_calculate_scores(stats)
        score_pth = eval_file.replace(".xlsx", "_score.json")
        result_pth = eval_file.replace(".xlsx", "_result.xlsx")
        dump(final_score_dict, score_pth)
        dump(pd.DataFrame(result), result_pth)
        return final_score_dict
