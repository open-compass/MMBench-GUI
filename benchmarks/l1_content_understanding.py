import os
import string
import pandas as pd
import json
import ast
import numpy as np
import warnings

from typing import Union
from vlmeval.dataset import (
    ImageBaseDataset,
    extract_answer_from_item,
    build_judge,
    DEBUG_MESSAGE,
)
from vlmeval.smp import (
    LMUDataRoot,
    get_logger,
    load,
    read_ok,
    toliststr,
    dump,
    gpt_key_set,
)
from vlmeval.utils import track_progress_rich


from ipdb import set_trace as st
from utils.import_utils import dynamic_import_function
from utils.misc import parser_answers_into_option

logger = get_logger("RUN")

if os.environ.get("L1_USE_WEIGHTED", 0) > 0:
    from .matrics import level1_calculate_scores_weighted as level1_calculate_scores
else:
    from .matrics import level1_calculate_scores_normal as level1_calculate_scores

JUDGE_MODEL = {
    "chatgpt-0125": "openai",
    "exact_matching": "exact_matching",
    "gpt-4-0125": "gpt4",
}
L1_SYSTEM_PROMPT_DEFAULT = """You are a GUI agent. You are given a screenshot of an application, a question and corresponding options. You need to choose one option as your answer for the question. Finally, you are ONLY allowed to return the single letter of your choice."""


class GUIContentUnderstanding(ImageBaseDataset):
    MODALITY = "IMAGE"
    TYPE = "GUI_Content_Understanding"
    DATASET_URL = {
        "MMBench-GUI_L1": "xxxxxxxxxxx.tsv",  # noqa
    }  # path
    DATASET_MD5 = {
        "MMBench-GUI_L1": "a9a4fd9eb1e4ae0355fee057bd55cbec",
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
            "easy",
            "medium",
            "hard",
        ], "mode must be one of ['all', 'easy', 'medium', 'hard']"
        self.dataset_name = "GUIContentUnderstanding"
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
        if "image_path" in data:
            data["image_path"] = [
                f"{platform}/{image_path}"
                for platform, image_path in zip(data["platform"], data["image_path"])
            ]

        if self.skip_noimg:
            data = data[~pd.isna(data["image_path"])]

        self.data = data

    def default_parse_response(self, response, meta):
        """
        Default parse function for the response.
        It should be overridden by the user if needed.
        """
        match = parser_answers_into_option(response)
        if match:
            return match
        else:
            return None

    def load_data(self):
        if not os.path.exists(
            os.path.join(LMUDataRoot(), "MMBench-GUI", "L1_annotations.json")
        ):
            url = self.DATASET_URL.get("MMBench-GUI_L1", None)
            file_md5 = (
                self.DATASET_MD5["MMBench-GUI_L1"]
                if "MMBench-GUI_L1" in self.DATASET_MD5
                else None
            )
        else:
            url = os.path.join(LMUDataRoot(), "MMBench-GUI", "L1_annotations.json")
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
            json_data = json.load(open(url, "r", encoding="utf-8"))
            return_data = []
            if self.mode == "all":
                index = 0
                for item in json_data:
                    all_data = {
                        "image_path": item["image_path"],
                        "image_size": item["image_size"],
                        "platform": item["platform"],
                        "app_name": item["app_name"],
                    }
                    for gr in item["groups"]:
                        gr.update(all_data, index=index)
                        index += 1
                        return_data.append(gr)
                return pd.DataFrame(return_data)
            elif self.mode == "easy":
                indices = 0
            elif self.mode == "medium":
                indices = 1
            elif self.mode == "hard":
                indices = 2
            else:
                raise ValueError(
                    f"Invalid mode: {self.mode}, must be one of ['all', 'easy', 'medium', 'hard']"
                )

            for item in json_data:
                all_data = {
                    "index": item["index"],
                    "image_path": item["image_path"],
                    "image_size": item["image_size"],
                    "platform": item["platform"],
                    "app_name": item["app_name"],
                }
                all_data.update(item["groups"][indices])
                return_data.append(all_data)
            return pd.DataFrame(return_data)

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(self.data.iloc[idx])

    def build_prompt(self, line, use_system=True, custom_system_prompt=None):

        if isinstance(line, int):
            line = self.data.iloc[line]
        tgt_path = self.dump_image(line)

        question = line["question"]
        options = {
            cand: line["options"][cand]
            for cand in string.ascii_uppercase
            if cand in line["options"] and not pd.isna(line["options"][cand])
        }
        options_prompt = "Options:\n"
        for key, item in options.items():
            options_prompt += f"{key}. {item}\n"
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        user_prompt = ""
        if hint is not None:
            user_prompt += f"Hint: {hint}\n"
        user_prompt += f"Question: {question}\n"
        if len(options):
            user_prompt += options_prompt
            user_prompt += "Please select the correct answer from the options above. \n"

        msgs = []
        # add system prompt
        if use_system:
            system_prompt = (
                L1_SYSTEM_PROMPT_DEFAULT
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

    def evaluate(self, eval_file, **judge_kwargs):
        # st()
        dataset = self.dataset_name
        nproc = judge_kwargs.pop("nproc", 4)

        # TODO:
        # Currently, we only support circular=False. Circular mode is coming soon.
        circular = False
        if "circular" in dataset.lower():
            data = load(eval_file)
            data["index"] = [int(x) for x in data["index"]]
            dump(data, eval_file)
            circular = True

        # determine the judge manner and corresponding model
        suffix = eval_file.split(".")[-1]
        model = judge_kwargs.get(
            "model", self.kwargs.get("match_mode", "exact_matching")
        )
        assert model in JUDGE_MODEL.keys()
        name_str = JUDGE_MODEL[model] if model in JUDGE_MODEL.keys() else model
        if model == "exact_matching":
            model = None
        elif gpt_key_set():
            model = build_judge(**judge_kwargs)
            if not model.working():
                warnings.warn(
                    "OPENAI API is not working properly, will use exact matching for evaluation"
                )
                warnings.warn(DEBUG_MESSAGE)
                model = None
        else:
            warnings.warn(
                "OPENAI_API_KEY is not set properly, will use exact matching for evaluation"
            )
            model = None
        result_file = eval_file.replace(f".{suffix}", f"_{name_str}_result.pkl")

        #
        data = load(eval_file)
        data = data.sort_values(by="index")
        data["prediction"] = [str(x) for x in data["prediction"]]
        meta = self.data
        meta_q_map = {x: y for x, y in zip(meta["index"], meta["question"])}
        data_map = {x: y for x, y in zip(data["index"], data["question"])}
        for k in data_map:
            assert (
                k in meta_q_map
            ), f"eval_file should be the same as or a subset of dataset {self.dataset_name}"

        if circular:
            result = eval_circular(
                model,
                data,
                meta,
                nproc,
                result_file,
                parse_function=self.parse_function,
                dataset_name=self.dataset_name,
            )
        else:
            result = eval_vanilla(
                model,
                data,
                meta,
                nproc,
                result_file,
                parse_function=self.parse_function,
                dataset_name=self.dataset_name,
            )

        eval_record = eval_file.replace(f".{suffix}", f"_{name_str}_result.{suffix}")
        dump(result, eval_record)
        result = load(eval_record)

        final_score_dict = level1_calculate_scores(result)
        score_file = eval_file.replace(f".{suffix}", "_scores.json")
        dump(final_score_dict, score_file)
        return final_score_dict


def eval_vanilla_single(model, item, parser_function=None, dataset_name=None):
    opt_num = len(ast.literal_eval(item["options"]))

    if model is None:
        assert (
            parser_function is not None
        ), "parser_function must be provided when model is None"
        opt = parser_function(item["prediction"], meta=item)
        if opt == item["GT"]:
            return dict(
                hit=1,
                parsed_prediction=f"{opt}",
                options_num=opt_num,
                log=f"Match Log: Parsed option is {opt} and groundtruth option is {item['GT']}.",
            )
        else:
            return dict(
                hit=0,
                parsed_prediction=f"{opt}",
                options_num=opt_num,
                log=f"Match Log: Parsed option is {opt} while groundtruth option is {item['GT']}.",
            )
    else:
        res = extract_answer_from_item(model, item, dataset_name=dataset_name)
        opt, match_log = res["opt"], res["log"]
        if opt == item["GT"]:
            return dict(
                hit=1,
                parsed_prediction=f"{opt}",
                options_num=opt_num,
                log=f"Match Log: {match_log}. ",
            )
        else:
            return dict(
                hit=0,
                parsed_prediction=f"{opt}",
                options_num=opt_num,
                log=f"Match Log: {match_log}. ",
            )


def eval_vanilla(
    model, data, meta, nproc, result_file, parse_function=None, dataset_name=None
):
    result = {}
    if os.path.exists(result_file):
        result = load(result_file)
    answer_map = {i: c for i, c in zip(meta["index"], meta["answer"])}

    data = data[data["index"].isin(answer_map)]
    data["GT"] = [answer_map[idx] for idx in data["index"]]
    items = []

    for i in range(len(data)):
        # Dealing with the normal part
        item = data.iloc[i]
        if item["index"] not in result:
            items.append(item)

    tups = [
        dict(
            model=model,
            item=x,
            parser_function=parse_function,
            dataset_name=dataset_name,
        )
        for x in items
    ]
    keys = [x["index"] for x in items]
    if len(tups):
        res = track_progress_rich(
            eval_vanilla_single,
            tups,
            nproc=nproc,
            chunksize=nproc,
            save=result_file,
            keys=keys,
        )
        result = load(result_file)
        for k, v in zip(keys, res):
            if k not in result:
                result[k] = v
    data["hit"] = [result[i]["hit"] for i in data["index"]]
    data["log"] = [result[i]["log"] for i in data["index"]]
    data["parsed_prediction"] = [result[i]["parsed_prediction"] for i in data["index"]]
    data["options_num"] = [result[i]["options_num"] for i in data["index"]]

    if "GT" in data:
        data.pop("GT")

    return data


def eval_circular(
    model, data, meta, nproc, result_file, parse_function=None, dataset_name=None
):
    """
    Coming soon!
    """
    pass
