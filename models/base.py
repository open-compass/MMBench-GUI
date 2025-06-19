
import torch
import openai

import numpy as np
from transformers import AutoConfig, AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from vlmeval.vlm.base import BaseModel
from vlmeval.api.base import BaseAPI
from typing import Callable, Union, Optional, List, Dict
from vlmeval.smp import get_gpu_memory, get_logger, encode_image_to_base64
from utils.import_utils import dynamic_import_function, dynamic_import
from utils.misc import get_api_info
from ipdb import set_trace as st

class APIModelWrapper(BaseAPI):
    def __init__(self, 
                 model_name_or_class, 
                 generate_cfg: dict=None, 
                 imp_type: str='openai', 
                 generate_function = None,
                 preprocess_function: Optional[Union[str, callable]] = '',
                 postprocess_function: Optional[Union[str, callable]] = '',
                 custom_prompt: Optional[Union[dict, None]] = None,
                 retry: int = 5,
                 wait: int = 5,
                 fail_msg='Failed to obtain answer via API.',
                 verbose=True,
                 **kwargs
            ):
        st()
        assert imp_type in ['openai', 'claude'], "imp_type must be one of ['openai', 'claude']"
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

        self.logger = get_logger('API')

        if len(kwargs):
            self.logger.info(f'APIModelWrapper received the following kwargs: {kwargs}')
            self.logger.info('Will try to use them as kwargs for preprocess and postprocess function. ')
        self.default_kwargs = kwargs

        self.system_prompt = kwargs.get('system_prompt', None)
        self.img_size = kwargs.get('img_size', -1)
        self.img_detail = kwargs.get('img_detail', 'low')

        if self.imp_type == 'openai':
            base_url, api_key, model_name = get_api_info(self.model_name_or_class)
            self.base_url = base_url
            self.api_key = api_key
            self.model_name = model_name
            self.model = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=60)
            self.is_api = True
            self.logger.info(f'Using API Base: {self.base_url}; API Key: {self.api_key}')
        elif self.imp_type == 'claude':
            pass
        else:
            raise ValueError(f"imp_type {self.imp_type} is not supported for APIModelWrapper.")
        
        if self.preprocess_function == '' or self.preprocess_function == None:
            self.pre_func = None
        else:
            self.pre_func = dynamic_import_function(self.preprocess_function) if '.' in self.preprocess_function else getattr(self.model, self.preprecess_function, None)
        if self.postprocess_function == '' or self.postprocess_function == None:
            self.post_func = None
        else:
            self.post_func = dynamic_import_function(self.postprocess_function) if '.' in self.postprocess_function else getattr(self.model, self.postprocess_function, None)
        
        if generate_function is not None and generate_function != '':
            self.logger.info('`generate_function` is useless in APIModelWrapper, `self.gen_func` will be ignored.')
        self.gen_func = None

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
        
    def _default_prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    ext = msg['value'].split('.')[-1].lower()
                    b64 = encode_image_to_base64(img, target_size=self.img_size)
                    img_struct = dict(url=f'data:image/{ext};base64,{b64}', detail=self.img_detail)
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list
        
    def default_prepocess_function(self, message, **kwargs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(message, list) and isinstance(message[0], dict)
        assert np.all(['type' in x for x in message]) or np.all(['role' in x for x in message]), message
        if 'role' in message[0]:
            assert message[-1]['role'] == 'user', message[-1]
            for item in message:
                input_msgs.append(dict(role=item['role'], content=self._default_prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self._default_prepare_itlist(message)))
        return input_msgs
        
    def generate_inner_openai(self, message, **kwargs):

        input_msgs = self.pre_func(message,  None, None, **kwargs) if isinstance(self.pre_func, Callable) else self.default_prepocess_function(message, **kwargs)
        # For API models, we only need to organize the messages with the correct format
        try:
            outputs = self.model.chat.completions.create(
                model=self.model_name,
                messages=input_msgs,
                **self.generate_cfg
            )
            if hasattr(outputs, 'choices'):
                ret_code = 200
            else:
                ret_code = 400
            response = outputs.choices[0].message.content if hasattr(outputs, 'choices') and len(outputs.choices) > 0 else outputs
            response = self.post_func(response, None, None, **kwargs) if isinstance(self.post_func, Callable) else response
        except Exception as e:
            ret_code = 500
            response = self.fail_msg
            outputs = e
                
        return ret_code, response, outputs

    def generate_inner(self, message, **kwargs):
        st()
        if self.imp_type == 'openai':
            # For API models, we only need to organize the messages with the correct format
            if not isinstance(message, list):
                raise ValueError("For API models, the input message should be a list of dictionaries.")
            return self.generate_inner_openai(message, **kwargs)




class LocalModelWrapper(BaseModel):
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
        assert imp_type in ['transformers', 'lmdeploy', 'vllm', 'api', 'custom'], "imp_type must be one of ['transformers', 'api', 'lmdeploy', 'vllm', 'custom']"

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
            self.is_api = False
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
        if self.gen_func is None and not self.is_api:
            raise ValueError(f"Generate function '{self.generate_function}' not found in the model when using a local model.")
        
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
        inputs = self.pre_func(message, self.model, self.processor, **kwargs) if isinstance(self.pre_func, Callable) else message

        #get outputs from the model with parameters of `generate_cfg`
        assert isinstance(inputs, dict) or isinstance(inputs, BatchFeature), f"Inputs should be a dictionary or transformers.feature_extraction_utils.BatchFeature, but got {type(inputs)}"
        outputs = self.gen_func(**inputs, **self.generate_cfg)

        #get final response with `postprocess_function`
        response = self.post_func(outputs, self.model, self.processor, **kwargs) if isinstance(self.post_func, Callable) else outputs
        
        return response
    

    def generate_inner(self, message, dataset=None, **kwargs):
        
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
        

