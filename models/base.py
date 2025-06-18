
import torch

from transformers import AutoConfig, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from vlmeval.vlm.base import BaseModel
from typing import Callable, Union, Optional, List, Dict
from vlmeval.smp import get_gpu_memory
from utils.import_utils import dynamic_import_function, dynamic_import



class ModelWrapper(BaseModel):
    def __init__(self, 
                 model_name_or_class, 
                 generate_cfg: dict=None, 
                 imp_type: str='transformers', 
                 generate_function = 'generate_inner',
                 preprocess_function: Optional[Union[str, callable]] = 'preprocess',
                 postprocess_function: Optional[Union[str, callable]] = 'postprocess',
                 custom_prompt: Optional[Union[dict, None]] = None,
                 **kwargs
            ):
        assert imp_type in ['transformers', 'lmdeploy', 'vllm', 'custom'], "imp_type must be one of ['transformers', 'lmdeploy', 'vllm', 'custom']"

        self.model_name_or_class = model_name_or_class
        self.generate_cfg = generate_cfg
        self.imp_type = imp_type
        self.generate_function = generate_function
        self.preprocess_function = preprocess_function
        self.postprocess_function = postprocess_function
        self.kwargs = kwargs
        self.custom_prompt = custom_prompt

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        assert max_gpu_mem > 0

        if imp_type == 'transformers':
            self.auto_config = AutoConfig.from_pretrained(model_name_or_class)
            MODEL_CLS = dynamic_import(self.auto_config.architectures[0])
            if MODEL_CLS is None:
                raise ValueError(f"Model class {self.auto_config.architectures[0]} not found in transformers.")
            
            self.model = MODEL_CLS.from_pretrained(
                model_name_or_class,
                torch_dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2",
            )

            self.processor = AutoProcessor.from_pretrained(model_name_or_class)
            self.model.eval()
        elif imp_type == 'lmdeploy':
            # TODO
            NotImplementedError("lmdeploy is not implemented yet. Please use transformers.")
        elif imp_type == 'vllm':
            #TODO
            NotImplementedError("vllm is not implemented yet. Please use transformers.")
        elif imp_type == 'custom':
            #TODO
            NotImplementedError("custom is not implemented yet. Please use transformers.")
        torch.cuda.empty_cache()

        self.pre_func = dynamic_import_function(self.preprocess_function) if '.' in self.preprocess_function else getattr(self.model, self.preprecess_function, None)
        self.post_func = dynamic_import_function(self.postprocess_function) if '.' in self.postprocess_function else getattr(self.model, self.postprocess_function, None)
        self.gen_func = getattr(self.model, self.generate_function, None)
        if self.gen_func is None:
            raise ValueError(f"Generate function '{self.generate_function}' not found in the model.")
        
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
        ValueError("You should implement your build function (for example `your_build_prompt_function`) in models.your_model_name.py and set `model.your_model_name.custom_prompt` with the format of `dict(dataset_name='your_build_prompt_function)`")

    def generate_inner_transformers(self, message, **kwargs):
        '''
         content of message:

            [
                dict(role='system', type='text', content='You are a helpful assistant.'),
                dict(role='user', type='image', content='path/to/image.jpg'),
                dict(role='user', type='text', content='What is in the image?'),
            ]
        '''
        kwargs.update(self.kwargs)

        # get inputs_ids and other values for generate function in `preprocess_function`
        # pre_func = dynamic_import_function(self.preprocess_function) if '.' in self.preprocess_function else getattr(self.model, self.preprecess_function, None)
        inputs = self.pre_func(message, self.model, self.processor, **kwargs) if isinstance(self.pre_func, Callable) else message

        #get outputs from the model with parameters of `generate_cfg`
        #call_func = getattr(self.model, self.generate_function, None)

        assert isinstance(inputs, dict) or isinstance(inputs, BatchFeature), f"Inputs should be a dictionary or transformers.feature_extraction_utils.BatchFeature, but got {type(inputs)}"
        outputs = self.gen_func(**inputs, **self.generate_cfg)

        #get final response with `postprocess_function`
        #post_func = dynamic_import_function(self.postprocess_function) if '.' in self.postprocess_function else getattr(self.model, self.postprocess_function, None)
        response = self.post_func(outputs, self.model, self.processor, **kwargs) if isinstance(self.post_func, Callable) else outputs
        
        return response


    def generate_inner(self, message, dataset=None, **kwargs):
        from ipdb import set_trace as st
        st()
        if self.imp_type == 'transformers':
            return self.generate_inner_transformers(message, dataset_name=dataset, **kwargs)
        elif self.imp_type == 'lmdeploy':
            # TODO
            pass
        elif self.imp_type == 'vllm':
            #TODO
            pass
        elif self.imp_type == 'custom':
            #TODO
            pass
        

