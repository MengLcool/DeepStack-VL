# login to huggingface

CKPT=${1:-"None"}
conv_template=${2:-"vicuna_v1"}
vistoken_patch_size=${3:-"None"}

eval_tasks=${eval_tasks:-"textvqa,chartqa,docvqa"}
master_port=${master_port:-"12345"}
GPUS=`nvidia-smi -L | wc -l`

echo $CKPT, $conv_template

accelerate launch --num_processes=$GPUS --main_process_port=${master_port} -m lmms_eval --model llava   \
    --model_args pretrained=$CKPT,conv_template=${conv_template} \
    --tasks $eval_tasks  --batch_size 1 --log_samples --log_samples_suffix lmms_eval --output_path $CKPT/logs/

