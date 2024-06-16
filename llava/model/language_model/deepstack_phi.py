#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, select_best_resolution, get_possible_resolutions_by_crops

from ..deepstack_arch import DeepstackMetaModel, DeepstackMetaForCausalLM, unpad_image
# from .custom_llama import CustomLlamaModel, CustomLLlamaForCausalLM
from .custom_phi import CustomPhi3Model, CustomPhi3ForCausalLM
from .phi3.configuration_phi3 import Phi3Config


class DeepstackPhiConfig(Phi3Config):
    model_type = "deepstack_phi"


class DeepstackPhiModel(DeepstackMetaModel, CustomPhi3Model):
    config_class = DeepstackPhiConfig

    def __init__(self, config: Phi3Config):
        super(DeepstackPhiModel, self).__init__(config)


class DeepstackPhiForCausalLM(CustomPhi3ForCausalLM, DeepstackMetaForCausalLM):
    config_class = DeepstackPhiConfig

    def __init__(self, config):
        super(DeepstackPhiForCausalLM, self).__init__(config)
        self.model = DeepstackPhiModel(config)
        # self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        # for anyres deepstack
        bestgrid_list: Optional[List[torch.Tensor]] = None,
        # for deepstack generation
        
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        image_id_pos, pre_encoded_image_features, local_features = None, None, None

        # FIXME: this a tmp implementation
        if getattr(self, 'kwargs_for_generation', None):
            kwargs_for_generation = self.kwargs_for_generation
            image_id_pos, local_features = kwargs_for_generation['image_id_pos'], kwargs_for_generation['local_features']
            self.kwargs_for_generation = None

        if self.need_process_deepstack(images):
            pre_encoded_image_features, local_features = self.process_deepstack(images, bestgrid_list)
            if input_ids is not None:
                image_id_pos = torch.where(input_ids==IMAGE_TOKEN_INDEX)

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                bestgrid_list=bestgrid_list,
                pre_encoded_image_features=pre_encoded_image_features
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_id_pos=image_id_pos,
            local_features = local_features,
            local_features_start = getattr(self.config, 'deepstack_start_id', 1),
            local_features_interval = getattr(self.config, 'deepstack_interval',1)
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        bestgrid_list: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        pre_encoded_image_features, local_features = None, None
        # preprocess layer stack

        # if images is not None and getattr(self.config, 'image_aspect_ratio', None) == 'deepstack' \
        #                       and getattr(self.config, 'mm_patch_merge_type', None) != 'spatial_unpad':
        if self.need_process_deepstack(images):
            if bestgrid_list is None:
                if '2xsample' in self.deepstack_args_for_ablation:
                    num_crops_sqrt = int((self.config.num_crops **0.5))
                    bestgrid_list = [(num_crops_sqrt, num_crops_sqrt) for x in image_sizes]
                else:
                    image_processor = self.get_vision_tower().image_processor
                    possible_resolutions = get_possible_resolutions_by_crops(self.config.num_crops, image_processor.crop_size)
                    bestres_list = [select_best_resolution(x, possible_resolutions) for x in image_sizes]
                    bestgrid_list = [(w//image_processor.crop_size['width'], h//image_processor.crop_size['height']) for (w,h) in bestres_list]

            pre_encoded_image_features, local_features = self.process_deepstack(images, bestgrid_list)
            if inputs is not None:
                image_id_pos = torch.where(inputs==IMAGE_TOKEN_INDEX)
            self.kwargs_for_generation = dict(image_id_pos=image_id_pos, local_features=local_features,)

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                bestgrid_list=bestgrid_list,
                pre_encoded_image_features=pre_encoded_image_features
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("deepstack_phi", DeepstackPhiConfig)
AutoModelForCausalLM.register(DeepstackPhiConfig, DeepstackMetaForCausalLM)
