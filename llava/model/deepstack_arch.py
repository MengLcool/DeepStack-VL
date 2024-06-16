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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
from .llava_arch import LlavaMetaModel, LlavaMetaForCausalLM, unpad_image

class DeepstackMetaModel(LlavaMetaModel):
    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', '') \
                    or getattr(config, 'is_anyres_deepstack', False):
                self.image_newline = nn.Parameter(
                    torch.zeros(config.hidden_size, dtype=self.dtype)
                )

    def initialize_vision_modules(self, model_args, fsdp=None, **kwargs):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class DeepstackMetaForCausalLM(LlavaMetaForCausalLM):

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None,
        bestgrid_list=None,
        pre_encoded_image_features=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if pre_encoded_image_features is not None:
            image_features = pre_encoded_image_features
        elif type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                if image_sizes is None:
                    assert bestgrid_list is not None
                    img_size = self.get_vision_tower().config.image_size
                    image_sizes = [(x[0]*img_size, x[1]*img_size) for x in bestgrid_list]
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            # (c, h, w)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        elif type(images) is list or isinstance(images[0], torch.Tensor) and images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        elif type(images) is list or isinstance(images[0], torch.Tensor) and images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1).to(self.device) for x in image_features]
        else:
            image_features = self.encode_images(images)

        if getattr(self.config, 'deepstack_args_for_ablation', None) is not None:
            deepstack_args_for_ablation = self.config.deepstack_args_for_ablation.split('+')
            if 'scaledown_clip' in deepstack_args_for_ablation:
                bs, l, c = image_features.shape
                l_sqrt = int(l**0.5)
                image_features = image_features.view(bs, l_sqrt, l_sqrt, c).permute(0,3,1,2).contiguous()
                image_features = nn.functional.avg_pool2d(image_features, (2,2), (2,2))
                image_features = image_features.flatten(2).permute(0,2,1).contiguous()

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            if max([len(x) for x in new_input_embeds]) > tokenizer_model_max_length:
                print(
                    f"WARNING: truncate: {max([len(x) for x in new_input_embeds])} vs. {tokenizer_model_max_length}."
                )

            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    # --------------- modification for deepstack ---------------#
    def process_plain_deepstack(self, images, bestgrid_list):
        num_crops = self.config.num_crops
        batch_size = len(images)
        # images, list of images of global-local views
        global_images = [x[:1] for x in images]
        local_images = [x[1:] for x in images]
        num_local_grids = [len(x) for x in local_images]

        if 'hires_local' in self.deepstack_args_for_ablation:
            assert not getattr(self.config, 'twoviews_unshare', None)
            global_images = torch.cat(global_images)
            local_images = torch.cat(local_images)
            bs, cc, hh, ww = global_images.shape
            local_images = local_images.view(bs, 2, 2, cc, hh, ww).permute(0,3,1,4,2,5).flatten(2,3).flatten(3,4)
            
            pre_encoded_image_features = self.encode_images(global_images)
            local_features_raw = self.encode_images(local_images)
            local_feature_list = local_features_raw
        else:
            concat_images = torch.cat(global_images + local_images, dim=0)
            vision_features = self.encode_images(concat_images)

            if getattr(self.config, 'twoviews_unshare', None):
                pre_encoded_image_features, local_features_raw = vision_features[0][:batch_size], vision_features[1][batch_size:]
            else:
                pre_encoded_image_features, local_features_raw = vision_features[:batch_size], vision_features[batch_size:]

            # TODO: support deepstack_with_anyres
            if self.deepstack_vit:
                return pre_encoded_image_features, None

            local_feature_list = local_features_raw.split(num_local_grids)

        deepstack_sampling = getattr(self.config, 'deepstack_sampling', '2d')

        if deepstack_sampling == '2d_scaledown':
            bs, l, c = pre_encoded_image_features.shape
            patch_size = int(l ** 0.5)
            pre_encoded_image_features = pre_encoded_image_features.view(bs, patch_size, patch_size, c).permute(0,3,1,2)
            local_features_raw = local_features_raw.view(-1, patch_size, patch_size, c).permute(0,3,1,2)
            pre_encoded_image_features = nn.functional.avg_pool2d(pre_encoded_image_features, (2,2), (2,2)).flatten(2).permute(0,2,1).contiguous()
            local_features_raw = nn.functional.avg_pool2d(local_features_raw, (2,2), (2,2)).flatten(2).permute(0,2,1).contiguous()
            deepstack_sampling = '2d'
            
        local_features = None

        local_features = []
        patch_size = int(pre_encoded_image_features.shape[1] ** 0.5)
        c = pre_encoded_image_features.shape[-1]
        for b_id, (local_feature, bestgrid) in enumerate(zip(local_feature_list, bestgrid_list)):
            grid_w, grid_h = bestgrid
            if 'hires_local' in self.deepstack_args_for_ablation:
                local_feature = local_feature.view(grid_h*patch_size, grid_h*patch_size, c) #(h, w, c)
            else:
                local_feature = local_feature.view(grid_h, grid_w, patch_size, patch_size, c)
                local_feature = local_feature.permute(0,2,1,3,4).flatten(0,1).flatten(1,2) #(h, w, c)
            # 2d uniform sampling on local feature
            if deepstack_sampling == '2d':
                local_feature = [local_feature[i::grid_h, j::grid_w] for i in range(grid_h) for j in range(grid_w)]
                local_feature = torch.stack(local_feature).flatten(1,2) # (num_local_view, seq_vis, c)
            elif deepstack_sampling == '1d':
                local_feature = local_feature.flatten(0,1)
                local_feature = [local_feature[i::grid_h * grid_w] for i in range(grid_h * grid_w)]
                local_feature = torch.stack(local_feature)
            elif deepstack_sampling == 'grid':
                local_feature = local_feature.view(grid_h, patch_size, grid_w, patch_size, c).permute(0,2,1,3,4).flatten(0,1).flatten(1,2).contiguous()
            else:
                raise NotImplementedError

            # dummy input for non-image sentences
            if grid_h * grid_w <= 1 and getattr(self.config, 'twoviews_unshare', None):
                local_feature = local_feature * 0
                pre_encoded_image_features[b_id] += local_feature[0]
            # pad to the max local_view
            if len(local_feature) < num_crops:
                local_feature = torch.cat([local_feature, local_feature.new_zeros(num_crops-len(local_feature), *local_feature.shape[1:])])
            local_features.append(local_feature)
        local_features = torch.stack(local_features, dim=1) # (num_crops, bs, seq_vis, c)

        return pre_encoded_image_features, local_features

    def process_image_grids(self, image_feature, bestgrid):
        base_image_feature = image_feature[0]
        image_feature = image_feature[1:]
        height = width = self.get_vision_tower().num_patches_per_side
        assert height * width == base_image_feature.shape[0]
        if self.config.image_aspect_ratio == 'deepstack':
            num_patch_width, num_patch_height = bestgrid
            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
        else:
            raise NotImplementedError
        # if 'unpad' in mm_patch_merge_type:
        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
        image_feature = image_feature.flatten(1, 2).flatten(2, 3) # (c, h, w)
        image_feature = torch.cat((
            image_feature,
            self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
        ), dim=-1)
        image_feature = image_feature.flatten(1, 2).transpose(0, 1)

        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
        return image_feature

    def process_deepstack_with_anyres(self, images, bestgrid_list):
        num_crops_inputs = [grid_w*grid_h+1 for (grid_w, grid_h) in bestgrid_list]
        num_local_views_for_deepstack = int(self.config.num_crops**0.5)**2
        num_crops_deepstack = [x*num_local_views_for_deepstack for x in num_crops_inputs]
        input_images = [x[:num_input] for x, num_input in zip(images, num_crops_inputs)]
        deepstack_images = [x[num_input:] for x, num_input in zip(images, num_crops_inputs)]

        concat_images = torch.cat(input_images + deepstack_images, dim=0)
        vision_features = self.encode_images(concat_images)
        num_crops_inputs_total = sum(num_crops_inputs)

        if getattr(self.config, 'twoviews_unshare', None):
            pre_encoded_image_features_raw, local_features_raw = vision_features[0][:num_crops_inputs_total], vision_features[1][num_crops_inputs_total:]
        else:
            pre_encoded_image_features_raw, local_features_raw = vision_features[:num_crops_inputs_total], vision_features[num_crops_inputs_total:]

        pre_encoded_image_features_raw = pre_encoded_image_features_raw.split(num_crops_inputs)
        local_features_raw = local_features_raw.split(num_crops_deepstack)
        
        pre_encoded_image_features = []
        local_features = []
        num_patches_per_side, c = self.get_vision_tower().num_patches_per_side, self.config.hidden_size
        n_deepstack_sqrt = int(num_local_views_for_deepstack ** 0.5)
        bestgrid_for_deepstack = (n_deepstack_sqrt, n_deepstack_sqrt)
        for global_feature_this, local_feature_this, bestgrid in zip(pre_encoded_image_features_raw, local_features_raw, bestgrid_list):
            global_feature_this = self.process_image_grids(global_feature_this, bestgrid)
            num_sub_grids = bestgrid[0]*bestgrid[1]+1 # global view + local views
            local_feature_this = local_feature_this.view(num_sub_grids, n_deepstack_sqrt, n_deepstack_sqrt, num_patches_per_side, num_patches_per_side, c)
            local_feature_this = local_feature_this.permute(0,1,3,2,4,5).flatten(1,2).flatten(2,3) #(x, h, w, c)
            local_feature_this = local_feature_this.view(-1, num_patches_per_side, n_deepstack_sqrt, num_patches_per_side, n_deepstack_sqrt, c)
            local_feature_this = local_feature_this.permute(2,4,0,1,3,5).flatten(0,1).flatten(2,3) #(n_stack, x, l, c)
            local_feature_this = [self.process_image_grids(local_feature_this_perlayer, bestgrid) for local_feature_this_perlayer in local_feature_this]

            pre_encoded_image_features.append(global_feature_this)
            local_features.append(local_feature_this)
        
        local_features = list(zip(*local_features))
        return pre_encoded_image_features, local_features

    def process_deepstack(self, images, bestgrid_list):
        if getattr(self.config, 'is_anyres_deepstack', False):
            pre_encoded_image_features,  local_features = self.process_deepstack_with_anyres(images, bestgrid_list)
        else:
            pre_encoded_image_features,  local_features =  self.process_plain_deepstack(images, bestgrid_list)
        
        if 'dummy_local' in self.deepstack_args_for_ablation:
            local_features = [pre_encoded_image_features] * len(local_features)
        if 'only_global_features' in self.deepstack_args_for_ablation:
            local_features = [pre_encoded_image_features]
            pre_encoded_image_features = pre_encoded_image_features * 0
        elif 'global_and_local' in self.deepstack_args_for_ablation:
            # local_features = [pre_encoded_image_features] + local_features
            local_features = [x for x in local_features]
            local_features = [pre_encoded_image_features] + local_features
            pre_encoded_image_features = pre_encoded_image_features * 0
        elif 'string_woglobal' in self.deepstack_args_for_ablation:
            local_features = [x for x in local_features]
            pre_encoded_image_features = torch.stack(local_features, dim=1).flatten(1,2)
            local_features = None
        elif 'string_withglobal' in self.deepstack_args_for_ablation:
            local_features = [pre_encoded_image_features] + [x for x in local_features]
            pre_encoded_image_features = torch.stack(local_features, dim=1).flatten(1,2)
            local_features = None

        return pre_encoded_image_features, local_features

    def need_process_deepstack(self, images):
        return images is not None \
            and (
                getattr(self.config, 'image_aspect_ratio', None) == 'deepstack'
            )

    @property
    def deepstack_vit(self):
        deepstack_args_for_ablation= getattr(self.config, 'deepstack_args_for_ablation', '')
        deepstack_args_for_ablation = deepstack_args_for_ablation.split('+')
        deepstack_vit = 'deepstack_vit' in deepstack_args_for_ablation
        return deepstack_vit

    @property
    def deepstack_args_for_ablation(self):
        deepstack_args_for_ablation = getattr(self.config, 'deepstack_args_for_ablation', None)
        deepstack_args_for_ablation = deepstack_args_for_ablation.split('+') if deepstack_args_for_ablation is not None else []
        return deepstack_args_for_ablation
