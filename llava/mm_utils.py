from PIL import Image
from io import BytesIO
import base64
import torch
import math
import ast

from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX


def get_value_from_config_kwargs(config, kwargs, key, default=None):
    return getattr(config, key, kwargs.get(key, default))

def get_possible_resolutions_by_crops(num_crops, crop_size):
    grid_pinpoints = []
    for i in range(1, num_crops+1):
        for j in range(1, num_crops // i + 1):
            if i == j == 1:
                # skip 336x336 image
                continue
            grid_pinpoints.append((i, j))

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    possible_resolutions = [(int(res[0] * crop_size['width']), int(res[1] * crop_size['height'])) for res in possible_resolutions]
    return possible_resolutions

def select_best_resolution(original_size, possible_resolutions):
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    return best_fit


def resize_and_pad_image(image, target_resolution):
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def divide_to_patches(image, patch_size):
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    return width // patch_size, height // patch_size


def process_anyres_image(image, processor,
                         num_crops=None, is_anyres_deepstack=False,
                         deepstack_args_for_ablation=None,
                         grid_pinpoints=None,
                         ):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """

    is_ori_anyres = grid_pinpoints is not None
    if grid_pinpoints is None:
        assert num_crops is not None
        grid_pinpoints = []
        for i in range(1, num_crops+1):
            for j in range(1, num_crops // i + 1):
                if i == j == 1:
                    # skip 336x336 image
                    continue
                grid_pinpoints.append((i, j))
        grid_pinpoints = [(int(res[0] * processor.crop_size['width']), int(res[1] * processor.crop_size['height'])) for res in grid_pinpoints]

    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)

    if isinstance(deepstack_args_for_ablation, str):
        deepstack_args_for_ablation = deepstack_args_for_ablation.split('+')

    if deepstack_args_for_ablation is not None and '2xsample' in deepstack_args_for_ablation:
        num_crops_sqrt = int((num_crops **0.5))
        best_resolution = (num_crops_sqrt*processor.crop_size['width'], num_crops_sqrt*processor.crop_size['height'])

    best_resolution_grid = (best_resolution[0] // processor.crop_size['width'], best_resolution[1] // processor.crop_size['height'])

    if is_ori_anyres:
        image_padded = resize_and_pad_image(image, best_resolution)
    else:
        if deepstack_args_for_ablation is not None and \
                ('inconsistent' in deepstack_args_for_ablation or 'anyres_unpad' in deepstack_args_for_ablation) :
            image_padded = resize_and_pad_image(image, best_resolution)
        else:
            image_padded = image.resize(best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size['height'])

    image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches

    if is_anyres_deepstack:
        upsample_scale = int(num_crops ** 0.5)
        image_padded_upsample = image.resize([x*upsample_scale for x in best_resolution])
        patches_upsample = divide_to_patches(image_padded_upsample, processor.crop_size['height'])

        # group upsampled grids for each grid
        indexes = torch.arange(upsample_scale*upsample_scale*best_resolution_grid[0]*best_resolution_grid[1])
        grid_w, grid_h = best_resolution_grid
        indexes = indexes.view(grid_h, upsample_scale, grid_w, upsample_scale)
        indexes = indexes.permute(0,2,1,3).flatten().tolist()
        patches_upsample = [patches_upsample[x] for x in indexes]

        image_original_resize_upsample = image.resize((processor.size['shortest_edge']*upsample_scale, processor.size['shortest_edge']*upsample_scale))
        patches_image_original_resize_upsample = divide_to_patches(image_original_resize_upsample, processor.crop_size['height'])
        image_patches = image_patches + patches_image_original_resize_upsample + patches_upsample

    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    return torch.stack(image_patches, dim=0), best_resolution_grid


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    is_anyres_deepstack = getattr(model_cfg, 'is_anyres_deepstack', False)

    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    elif image_aspect_ratio == "deepstack":
        for image in images:
            image, _ = process_anyres_image(image, image_processor,
                                            num_crops=model_cfg.num_crops, is_anyres_deepstack=is_anyres_deepstack,
                                            deepstack_args_for_ablation=getattr(model_cfg, 'deepstack_args_for_ablation', None))
            new_images.append(image)
    elif image_aspect_ratio == "anyres":
        for image in images:
            image, _ = process_anyres_image(image, image_processor, grid_pinpoints=model_cfg.image_grid_pinpoints)
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            truncated_output_ids = output_ids[0, -keyword_id.shape[0]:]
            if torch.equal(truncated_output_ids, keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
