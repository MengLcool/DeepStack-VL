import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers.models.clip.modeling_clip import BaseModelOutputWithPooling, BaseModelOutput
from transformers import AutoProcessor, AutoModel


def custom_embedding(embedding_module, pixel_values: torch.FloatTensor) -> torch.Tensor:
    batch_size = pixel_values.shape[0]
    target_dtype = embedding_module.patch_embedding.weight.dtype
    patch_embeds = embedding_module.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
    patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

    class_embeds = embedding_module.class_embedding.expand(batch_size, 1, -1)
    embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
    pos_embed = embedding_module.position_embedding(embedding_module.position_ids)
    
    # ---------------- pos embed interpolate --------------------
    class_pos_embed = pos_embed[:, 0]
    patch_pos_embed = pos_embed[:, 1:]
    dim = patch_pos_embed.shape[-1]
    
    patch_embeds_HW = int(patch_embeds.shape[1]**0.5)
    kwargs = {}
    kwargs["size"] = (patch_embeds_HW, patch_embeds_HW)
    N = patch_pos_embed.shape[1] 
    M = int(N ** 0.5)  # Recover the number of patches in each dimension
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, M, M, dim).permute(0, 3, 1, 2),
        mode="bicubic",
        antialias=False,
        **kwargs,
    )
    patch_pos_embed = patch_pos_embed.flatten(2,3).permute(0,2,1)
    pos_embed = torch.cat([class_pos_embed, patch_pos_embed], dim=1)
    # ---------------- end of pos embed interpolate --------------------

    embeddings = embeddings + pos_embed
    return embeddings

class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False, **kwargs):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.unfreeze_mm_vision_tower= getattr(args, 'unfreeze_mm_vision_tower', False)

        # args for deepstack vit
        deepstack_args_for_ablation= getattr(args, 'deepstack_args_for_ablation', '')
        deepstack_args_for_ablation = deepstack_args_for_ablation.split('+') if deepstack_args_for_ablation is not None else []
        self.deepstack_vit = 'deepstack_vit' in deepstack_args_for_ablation
        if self.deepstack_vit:
            lsvit_start_layer = [x for x in deepstack_args_for_ablation if x.startswith('lsvit_start_layer')]
            assert len(lsvit_start_layer) <= 1
            if len(lsvit_start_layer) == 0:
                self.lsvit_start_layer = None
            elif len(lsvit_start_layer) == 1:
                self.lsvit_start_layer = int(lsvit_start_layer[0].split('_')[-1])
            else:
                raise NotImplementedError
        self.num_crops = getattr(args, 'num_crops', 4)

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        if not self.unfreeze_mm_vision_tower:
            self.vision_tower.requires_grad_(False)

        if self.deepstack_vit and self.lsvit_start_layer >= len(self.vision_tower.vision_model.encoder.layers):
            # FIXME: force use the last layer for LLM
            from transformers.models.clip.modeling_clip import CLIPEncoderLayer
            self.post_encoder = nn.ModuleList([CLIPEncoderLayer(self.vision_tower.config) for _ in range(5)])
            self.select_layer = -1

        self.is_loaded = True

    def custom_feature_forward(self, x, output_hidden_states=True, **kwargs):
        if not self.deepstack_vit :
            return self.vision_tower(x, output_hidden_states=output_hidden_states)

        pixel_values = x
        output_attentions = False
        return_dict = True
        
        vision_model = self.vision_tower.vision_model
        ncrops_sqrt = int(self.num_crops ** 0.5)
        assert x.shape[0] % (self.num_crops+1) == 0
        bs = x.shape[0] // (self.num_crops+1)
        pixel_values_global, pixel_values_local = pixel_values[:bs], pixel_values[bs:]
        h, w = pixel_values_global.shape[-2:]

        hidden_states = vision_model.embeddings(pixel_values_global)
        hidden_states = vision_model.pre_layrnorm(hidden_states) # (bs,l,c)
        
        if self.deepstack_vit:
            hidden_states_local = vision_model.embeddings(pixel_values_local)
            hidden_states_local = vision_model.pre_layrnorm(hidden_states_local) # (bs,l,c)
            if self.lsvit_start_layer is None:
                local_features_start = len(vision_model.encoder.layers) // 2
            else:
                local_features_start = self.lsvit_start_layer
            hidden_states = torch.cat([hidden_states, hidden_states_local], dim=0)
            local_features = None
        else:
            # TODO: support pos interpolation
            pixel_values_local = pixel_values_local.view(bs, ncrops_sqrt, ncrops_sqrt, 3, h, w)
            pixel_values_local = pixel_values_local.permute(0,3,1,4,2,5).flatten(2,3).flatten(3,4).contiguous()
            hidden_states_local = custom_embedding(vision_model.embeddings, pixel_values_local)
            hidden_states_local = vision_model.pre_layrnorm(hidden_states_local) # (bs,HH*WW,c)
            local_features_start = 1
            h = w = int((hidden_states.shape[1]-1)**0.5)
            local_features = hidden_states_local[:, 1:] #(bs,HH*WW,c)
            local_features = local_features.view(bs, h, ncrops_sqrt, w, ncrops_sqrt, -1)
            local_features = local_features.permute(2,4,0,1,3,5).contiguous().flatten(0,1).flatten(2,3) #(ncrop,bs,l,c)


        # --------------- start of encoder --------------#
        attention_mask = None
        causal_attention_mask = None
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(vision_model.encoder.layers):
            
            if idx== local_features_start and self.deepstack_vit:
                hidden_states, hidden_states_local = hidden_states[:bs], hidden_states[bs:]
                h = w = int((hidden_states.shape[1]-1)**0.5)
                local_features = hidden_states_local[:, 1:] #(bs,HH*WW,c)
                local_features = local_features.view(bs, ncrops_sqrt, ncrops_sqrt, h, w, -1)
                local_features = local_features.permute(0,1,3,2,4,5).contiguous().view(bs, h, ncrops_sqrt, w, ncrops_sqrt, -1)
                local_features = local_features.permute(2,4,0,1,3,5).flatten(0,1).flatten(2,3) #(ncrop,bs,l,c)
                # local_features = local_features.permute(1,3,0,2,4,5).contiguous().flatten(0,1).flatten(2,3) #(ncrop,bs,l,c)

            if local_features is not None and idx >= local_features_start and idx < local_features_start + len(local_features):
                hidden_states[:, 1:] += local_features[idx-local_features_start]

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if vision_model.encoder.gradient_checkpointing and vision_model.encoder.training:
                layer_outputs = vision_model.encoder._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        
        # --------------- end of encoder --------------#

        if self.deepstack_vit and self.lsvit_start_layer >= len(self.vision_tower.vision_model.encoder.layers):

            for idx_this, encoder_layer in enumerate(self.post_encoder):
                idx = idx_this + len(self.vision_tower.vision_model.encoder.layers)

                if idx== local_features_start and self.deepstack_vit:
                    hidden_states, hidden_states_local = hidden_states[:bs], hidden_states[bs:]
                    h = w = int((hidden_states.shape[1]-1)**0.5)
                    local_features = hidden_states_local[:, 1:] #(bs,HH*WW,c)
                    local_features = local_features.view(bs, ncrops_sqrt, ncrops_sqrt, h, w, -1)
                    local_features = local_features.permute(0,1,3,2,4,5).contiguous().view(bs, h, ncrops_sqrt, w, ncrops_sqrt, -1)
                    local_features = local_features.permute(2,4,0,1,3,5).flatten(0,1).flatten(2,3) #(ncrop,bs,l,c)
                    # local_features = local_features.permute(1,3,0,2,4,5).contiguous().flatten(0,1).flatten(2,3) #(ncrop,bs,l,c)

                if local_features is not None and idx >= local_features_start and idx < local_features_start + len(local_features):
                    hidden_states[:, 1:] += local_features[idx-local_features_start]

                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)

                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
        
        encoder_outputs = BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

        last_hidden_state = encoder_outputs[0]
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = vision_model.post_layernorm(pooled_output)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def forward(self, images, **kwargs):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.custom_feature_forward(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.custom_feature_forward(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2

class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
