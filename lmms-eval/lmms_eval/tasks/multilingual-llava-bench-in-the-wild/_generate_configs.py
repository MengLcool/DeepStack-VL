from datasets import load_dataset

# dataset = load_dataset("gagan3012/multilingual-llava-bench")

configs = ['arabic', 'bengali', 'chinese', 'french', 'hindi', 'japanese', 'russian', 'spanish', 'urdu']

for config in configs:
    yaml_output = f"""
    dataset_path: "gagan3012/multilingual-llava-bench"
    dataset_kwargs:
        config: {config}
        token: True
    task: "llava_in_the_wild_{config}"
    test_split: train
    output_type: generate_until
    doc_to_visual: !function utils.llava_doc_to_visual
    doc_to_text: !function utils.llava_doc_to_text
    doc_to_target: "gpt_answer"
    generation_kwargs:
    until:
        - "ASSISTANT:"
    image_aspect_ratio: original
    max_new_tokens: 1024
    temperature: 0
    top_p: 0
    num_beams: 1
    do_sample: false
    process_results: !function utils.llava_process_results
    metric_list:
    - metric: gpt_eval_llava_all
        aggregation: !function utils.llava_all_aggregation
        higher_is_better: true
    - metric: gpt_eval_llava_conv
        aggregation: !function utils.llava_conv_aggregation
        higher_is_better: true
    - metric: gpt_eval_llava_detail
        aggregation: !function utils.llava_detail_aggregation
        higher_is_better: true
    - metric: gpt_eval_llava_complex
        aggregation: !function utils.llava_complex_aggregation
        higher_is_better: true
    metadata:
    version: 0.0
    gpt_eval_model_name: "gpt-4-0613"
    model_specific_prompt_kwargs:
    default:
        pre_prompt: ""
        post_prompt: ""
    """

    with open(f"{config}_llava_in_the_wild.yaml", "w") as f:
        f.write(yaml_output)

# Path: _generate_configs.py