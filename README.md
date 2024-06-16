# <i>DeepStack</i>: Deeply Stacking Visual Tokens is Surprisingly Simple and Effective for LMMs

### [Project Page](https://deepstack-vl.github.io/) | [Paper (ArXiv)](https://arxiv.org/abs/2406.04334) | [Model Zoo](docs/MODEL_ZOO.md)

In this work, we introduce DeepStack, a simple and effective strategy for providing informative visual information by stacking visual tokens from bottom to top, maintaining the same visual context length.

## ⏳ : News
+ [6/16] 🔥 Training and evaluation codes are released.
+ [6/06] 🔥 We released DeepStack. We propose to infuses visual tokens into different transformer layers without increasing the visual context length. 

## DeepStack LMM
![teaser](assets/deepstack_teaser.png)

## Contents
- [Install](#install)
- [DeepStack Weights](#deepstack-weights)
- [Train](#train)
- [Evaluation](#evaluation)
- [Model Zoo](docs/MODEL_ZOO.md)
- [Architecture](#architecture)
- [Visualization](#visualization)

## Install


1. Clone this repository and install packages
```bash
git clone git@github.com:MengLcool/DeepStack-VL.git
cd DeepStack-VL
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

2. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

3. Install additional packages for llms-eval evaluation
```
cd lmms-eval/
pip install -e .
cd ../

pip install git+https://github.com/huggingface/huggingface_hub
huggingface-cli login --token your/hf/tokens
```

## Train
```

```

## Evaluation
We provide a script to use lmms eval for evaluation.
Your can use eval_tasks to specify the evaluation tasks. 
```
# specify evaluation tasks
export eval_tasks=textvqa,chartqa,docvqa

# for ckpts with vicuna as LLM
bash scripts/eval_lmms.sh $CKPT vicuna_v1

# for ckpts with phi-3 as LLM
bash scripts/eval_lmms.sh $CKPT phi3_instruct
```

## Architecture
![arch](assets/deepstack_vl.png)
The framework of DeepStack is quite simple: the main innovation lies in the DeepStack strategy that infuses visual tokens into different layers. 

DeepStack-L: DeepStack for LLMs. Given an input image, we feed the tokens extracted from the low-resolution version to the input layer of LLM. Considering the 2D nature of images, we extra the neighbors from the high-resolution version and reorganize them into DeepStack, which are then fed to the consequent layers in LLMs. 

DeepStack-V: DeepStack for ViTs. We apply similar sampling strategy but feed the visual tokens into the ViT layers of vision encoder.


## Visualization
![example](assets/visualization.png)


## Citation

If you find our work helpful for your research, please consider citing the following BibTeX entry.   

```bibtex
@misc{meng2024deepstack,
      title={DeepStack: Deeply Stacking Visual Tokens is Surprisingly Simple and Effective for LMMs}, 
      author={Meng, Lingchen and Yang, Jianwei and Tian, Rui and Dai, Xiyang and Wu, Zuxuan and Gao, Jianfeng and Jiang, Yu-Gang}
      publisher={arXiv:2406.04334},
      year={2024},
}

