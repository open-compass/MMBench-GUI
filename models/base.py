import torch
import openai
import os
import inspect

import numpy as np
from transformers import AutoConfig
from transformers.feature_extraction_utils import BatchFeature
from vlmeval.vlm.base import BaseModel
from vlmeval.api.base import BaseAPI
from typing import Callable, Union, Optional, List, Dict
from vlmeval.smp import get_gpu_memory, get_logger, encode_image_to_base64
from utils.import_utils import dynamic_import_function, dynamic_import
from utils.misc import get_api_info, ensure_image_url
from ipdb import set_trace as st

logger = get_logger("MMBench-GUI")


class LazyAutoProcessor:
    """
    A lazily-loaded AutoProcessor to prevent tokenizer-related deadlocks
    when using multiprocessing (e.g., DataLoader with num_workers > 0).

    This class delays the actual loading of the processor until the first
    time it is used (either by attribute access or by calling it as a function).

    Example:
        processor = LazyAutoProcessor("Qwen/Qwen2.5-VL")
        inputs = processor(text="你好", images=image, return_tensors="pt")
        output_text = processor.batch_decode(generated_ids)
    """

    def __init__(self, model_name_or_path: str, **kwargs):
        """
        Args:
            model_name_or_path (str): Model name or local path used by AutoProcessor.
            kwargs: Optional kwargs passed to `from_pretrained`, such as cache_dir, revision, etc.
        """
        self.model_name_or_path = model_name_or_path
        self._processor = None
        self._kwargs = kwargs

    def _load(self):
        """
        Internal lazy loader. Actually loads the processor when first accessed.
        """
        if self._processor is None:
            # import os, inspect, time

            # pid = os.getpid()
            # ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # ddp_ready = os.environ.get("DDP_READY", "NOT SET")
            # print("=" * 60)
            # print(f"[{ts}] LazyAutoProcessor triggered in PID {pid}")
            # print(f"DDP_READY = {ddp_ready}")
            # print("Call stack:")
            # for frame in inspect.stack()[1:6]:
            #     print(f"→ {frame.filename}:{frame.lineno} in {frame.function}")
            #     if frame.code_context:
            #         print(f"   ↳ {frame.code_context[0].strip()}")
            # print("=" * 60)
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.model_name_or_path, **self._kwargs
            )

    def __getattr__(self, attr):
        """
        Allows access to any attribute or method of the real processor.
        Will automatically trigger loading on first access.
        """
        self._load()
        return getattr(self._processor, attr)

    def __call__(self, *args, **kwargs):
        """
        Allows the LazyAutoProcessor to be called like a function,
        e.g., processor(text=..., images=..., return_tensors="pt")
        """
        self._load()
        return self._processor(*args, **kwargs)


class APIModelWrapper(BaseAPI):
    def __init__(
        self,
        model_name_or_class,
        generate_cfg: dict = None,
        imp_type: str = "openai",
        generate_function=None,
        preprocess_function: Optional[Union[str, callable]] = None,
        postprocess_function: Optional[Union[str, callable]] = None,
        custom_prompt: Optional[Union[dict, None]] = None,
        retry: int = 5,
        wait: int = 5,
        fail_msg="Failed to obtain answer via API.",
        verbose=True,
        **kwargs,
    ):
        assert imp_type in [
            "openai",
            "claude",
        ], "imp_type must be one of ['openai', 'claude']"
        self.wait = wait
        self.retry = retry
        self.verbose = verbose
        self.fail_msg = fail_msg

        self.model_name_or_class = model_name_or_class
        self.generate_cfg = generate_cfg
        self.imp_type = imp_type
        self.generate_function = generate_function
        self.preprocess_function = preprocess_function
        self.postprocess_function = postprocess_function
        self.custom_prompt = custom_prompt

        self.logger = get_logger("API")

        if len(kwargs):
            self.logger.info(f"APIModelWrapper received the following kwargs: {kwargs}")
            self.logger.info(
                "Will try to use them as kwargs for preprocess and postprocess function. "
            )
        self.default_kwargs = kwargs

        self.system_prompt_mode = kwargs.get("system_prompt", None)
        self.system_prompt = (
            ""
            if self.system_prompt_mode in ["model_default", "benchmark_default"]
            else self.system_prompt_mode
        )
        self.img_size = kwargs.get("img_size", -1)
        self.img_detail = kwargs.get("img_detail", "low")

        if self.imp_type == "openai":
            base_url, api_key, model_name = get_api_info(self.model_name_or_class)
            self.base_url = base_url
            self.api_key = api_key
            self.model_name = model_name
            self.model = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=60)
            self.is_api = True
            self.logger.info(
                f"Using API Base: {self.base_url}; API Key: {self.api_key}"
            )
        elif self.imp_type == "claude":
            pass
        else:
            raise ValueError(
                f"imp_type {self.imp_type} is not supported for APIModelWrapper."
            )

        if self.preprocess_function == "" or self.preprocess_function == None:
            self.pre_func = self.default_preprocess_function
        else:
            assert (
                "." in self.preprocess_function
            ), f"The string of `preprocess_function` in your config should contains at least one `.`, for example `models.your_api_model.preprocess_function`"
            self.pre_func = dynamic_import_function(self.preprocess_function)

        if self.postprocess_function == "" or self.postprocess_function == None:
            self.post_func = self.default_postprocess_function
        else:
            assert (
                "." in self.postprocess_function
            ), f"The string of `postprocess_function` in your config should contains at least one `.`, for example `models.your_api_model.postprocess_function`"
            self.post_func = dynamic_import_function(self.postprocess_function)

        if generate_function is not None and generate_function != "":
            self.logger.info(
                "`generate_function` is useless in APIModelWrapper, `self.gen_func` will be ignored."
            )
        self.gen_func = None

        self.use_custom_prompt_status = False  # Used to process the situation: when you build prompt by yourself, system (if you provide) will be preserved in messages

    def use_custom_prompt(self, dataset):
        if self.custom_prompt is None:
            self.build_prompt = self.default_build_prompt
            self.use_custom_prompt_status = False
            return False
        if dataset in self.custom_prompt:
            if self.custom_prompt[dataset] is None:
                self.build_prompt = self.default_build_prompt
                self.use_custom_prompt_status = False
                return False
            self.build_prompt = dynamic_import_function(self.custom_prompt[dataset])
            self.use_custom_prompt_status = True
            return True
        else:
            self.build_prompt = self.default_build_prompt
            self.use_custom_prompt_status = False
            return False

    def default_build_prompt(self, **kwargs):
        ValueError(
            "You should implement your build function (for example `your_build_prompt_function`) in models.your_model_name.py and set `model.your_model_name.custom_prompt` with the format of `dict(dataset_name='your_build_prompt_function)`"
        )

    def _default_prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x["type"] == "image" for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg["type"] == "text":
                    content_list.append(dict(type="text", text=msg["value"]))
                elif msg["type"] == "image":
                    from PIL import Image

                    img = Image.open(msg["value"])
                    ext = msg["value"].split(".")[-1].lower()
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    img_struct = dict(
                        url=f"data:image/{ext};base64,{b64}", detail=self.img_detail
                    )
                    content_list.append(dict(type="image_url", image_url=img_struct))
        else:
            assert all([x["type"] == "text" for x in inputs])
            text = "\n".join([x["value"] for x in inputs])
            content_list = [dict(type="text", text=text)]
        return content_list

    def default_preprocess_function(self, message, model, processor, **kwargs):
        input_msgs = []
        system_index = [
            i for i, x in enumerate(message) if x.get("role", None) == "system"
        ]

        if len(system_index) > 0:
            pass
        elif self.system_prompt != "" and self.system_prompt is not None:
            input_msgs.append(dict(role="system", content=self.system_prompt))

        assert isinstance(message, list) and isinstance(message[0], dict)
        assert np.all(["type" in x for x in message]) or np.all(
            ["role" in x for x in message]
        ), message

        if "role" in message[0]:
            assert message[-1]["role"] == "user", message[-1]
            for item in message:
                input_msgs.append(
                    dict(
                        role=item["role"], content=self._default_prepare_itlist([item])
                    )
                )
        else:
            input_msgs.append(
                dict(role="user", content=self._default_prepare_itlist(message))
            )
        return input_msgs

    def default_postprocess_function(self, message, model, processor, **kwargs):
        # Due to various response format when calling api model, we directly return the response as our default implementation.
        # Thus, the desired output used for calculating matrics should be processed by `parse_gui_xxxxxx` functions implemented in your model file.
        return message

    def generate_inner_openai(self, message, **kwargs):

        input_msgs = self.pre_func(message, None, None, **kwargs)
        # For API models, we only need to organize the messages with the correct format
        try:
            outputs = self.model.chat.completions.create(
                model=self.model_name, messages=input_msgs, **self.generate_cfg
            )
            if hasattr(outputs, "choices"):
                ret_code = 0
            else:
                ret_code = 400
            response = (
                outputs.choices[0].message.content
                if hasattr(outputs, "choices") and len(outputs.choices) > 0
                else outputs
            )
            response = self.post_func(response, None, None, **kwargs)
        except Exception as e:
            ret_code = 500
            response = self.fail_msg
            outputs = e

        return ret_code, response, outputs

    def generate_inner(self, message, **kwargs):
        # st()
        if self.imp_type == "openai":
            # For API models, we only need to organize the messages with the correct format
            if not isinstance(message, list):
                raise ValueError(
                    "For API models, the input message should be a list of dictionaries."
                )
            if "dataset" in kwargs:
                kwargs["dataset_name"] = kwargs["dataset"]
                kwargs.pop("dataset")
            return self.generate_inner_openai(message, **kwargs)

    def preprocess_message_with_role(self, message):
        system_prompt = ""
        new_message = []
        # st()
        if self.use_custom_prompt_status:
            for data in message:
                assert isinstance(data, dict)
                role = data["role"]
                if (
                    role == "system"
                ):  # we preserve system prompt when using custom prompt
                    system_prompt += data["value"] + "\n"
                    new_message.append(
                        {
                            "role": data["role"],
                            "type": data["type"],
                            "value": system_prompt,
                        }
                    )
                else:
                    new_message.append(data)
        else:
            for data in message:
                assert isinstance(data, dict)
                role = data["role"]
                if role == "system":
                    system_prompt += data["value"] + "\n"
                else:
                    new_message.append(data)

        if system_prompt != "":
            if self.system_prompt_mode == "benchmark_default":
                self.system_prompt = system_prompt
            elif self.system_prompt_mode == "model_default":
                self.system_prompt = ""
            else:
                self.system_prompt = system_prompt

        return new_message


class LocalModelWrapper(BaseModel):
    def __init__(
        self,
        model_name_or_class,
        generate_cfg: dict = None,
        imp_type: str = "transformers",
        generate_function="generate",
        preprocess_function: Optional[
            Union[str, callable]
        ] = "models.base.default_preprocess_function",
        postprocess_function: Optional[
            Union[str, callable]
        ] = "models.base.default_postprocess_function",
        custom_prompt: Optional[Union[dict, None]] = None,
        **kwargs,
    ):
        assert imp_type in [
            "transformers",
            "lmdeploy",
            "vllm",
            "api",
            "custom",
        ], "imp_type must be one of ['transformers', 'api', 'lmdeploy', 'vllm', 'custom']"
        self.model_name_or_class = model_name_or_class
        self.generate_cfg = generate_cfg
        self.imp_type = imp_type
        self.generate_function = generate_function
        self.preprocess_function = preprocess_function
        self.postprocess_function = postprocess_function
        self.default_kwargs = kwargs
        self.custom_prompt = custom_prompt
        self.system_prompt = kwargs.get("system_prompt", None)
        self.img_size = kwargs.get("img_size", -1)
        self.img_detail = kwargs.get("img_detail", "low")

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        if imp_type == "transformers":
            assert os.path.exists(
                model_name_or_class
            ), f"Model path {model_name_or_class} does not exist. Please check your config file."
            assert os.path.exists(
                os.path.join(model_name_or_class, "config.json")
            ), f"`config.json` of model {model_name_or_class} does not exist. Please check your checkpoint."

            self.auto_config = AutoConfig.from_pretrained(model_name_or_class)
            try:
                MODEL_CLS = dynamic_import(self.auto_config.architectures[0])
            except:
                raise ImportError(
                    f"Model class {self.auto_config.architectures[0]} not found in transformers. Please check your config file or install the required package."
                )
            if MODEL_CLS is None:
                raise ValueError(
                    f"Model class {self.auto_config.architectures[0]} not found in transformers."
                )
            self.model = MODEL_CLS.from_pretrained(
                model_name_or_class,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2",
            )
            self.processor = LazyAutoProcessor(model_name_or_class)

            self.model.eval()
            self.is_api = False
        elif imp_type == "lmdeploy":
            # TODO
            NotImplementedError(
                "lmdeploy is not implemented yet. Please use transformers."
            )
        elif imp_type == "vllm":
            # TODO
            NotImplementedError("vllm is not implemented yet. Please use transformers.")
        elif imp_type == "custom":
            # TODO
            NotImplementedError(
                "custom is not implemented yet. Please use transformers."
            )
        torch.cuda.empty_cache()

        self.pre_func = (
            dynamic_import_function(self.preprocess_function)
            if "." in self.preprocess_function
            else getattr(
                self.model, self.preprecess_function, self.default_preprocess_function
            )
        )
        self.post_func = (
            dynamic_import_function(self.postprocess_function)
            if "." in self.postprocess_function
            else getattr(
                self.model, self.postprocess_function, self.default_postprocess_function
            )
        )
        self.gen_func = getattr(self.model, self.generate_function, None)
        if self.gen_func is None and not self.is_api:
            raise ValueError(
                f"Generate function '{self.generate_function}' not found in the model when using a local model."
            )

    def use_custom_prompt(self, dataset):
        if self.custom_prompt is None:
            self.build_prompt = self.default_build_prompt
            return False
        if dataset in self.custom_prompt:
            if self.custom_prompt[dataset] is None:
                self.build_prompt = self.default_build_prompt
                return False
            self.build_prompt = dynamic_import_function(self.custom_prompt[dataset])
            return True
        else:
            self.build_prompt = self.default_build_prompt
            return False

    def default_build_prompt(self, **kwargs):
        ValueError(
            "You should implement your build function (for example `your_build_prompt_function`) in models.your_model_name.py and set `model.your_model_name.custom_prompt` with the format of `dict(dataset_name='your_build_prompt_function)`"
        )

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

    def default_postprocess_function(self, message, model, processor, **kwargs):
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().numpy()
        out = processor.batch_decode(outputs, skip_special_tokens=True)

        response = out[0]
        print(f"\033[32m{response}\033[0m")
        return response

    def generate_inner_transformers(self, message, **kwargs):
        """
        content of message:

           [
               dict(role='system', type='text', value='You are a helpful assistant.'),
               dict(role='user', type='image', value='path/to/image.jpg'),
               dict(role='user', type='text', value='What is in the image?'),
           ]
        """

        kwargs.update(self.default_kwargs)

        # get inputs_ids and other values for generate function in `preprocess_function`
        inputs = self.pre_func(message, self.model, self.processor, **kwargs)

        assert isinstance(inputs, dict) or isinstance(
            inputs, BatchFeature
        ), f"Inputs should be a dictionary or transformers.feature_extraction_utils.BatchFeature, but got {type(inputs)}"
        # get outputs from the model with parameters of `generate_cfg`
        outputs = self.gen_func(**inputs, **self.generate_cfg)

        # get final response with `postprocess_function`
        response = self.post_func(outputs, self.model, self.processor, **kwargs)
        assert isinstance(
            response, str
        ), f"The output of your postprocess function must be a string, not a {type(response)}"
        return response

    def generate_inner(self, message, dataset=None, **kwargs):

        if self.imp_type == "transformers":
            return self.generate_inner_transformers(
                message, dataset_name=dataset, **kwargs
            )
        elif self.imp_type == "lmdeploy":
            # TODO
            pass
        elif self.imp_type == "vllm":
            # TODO
            pass
        elif self.imp_type == "custom":
            # TODO
            pass


def simple_prepare_content(inputs, processor, **kwargs):
    """
    inputs list[dict[str, str]], each dict has keys: ['type', 'value']
    """

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
