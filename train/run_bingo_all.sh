# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

set -x

export HYDRA_FULL_ERROR=1

# dataset config
DATASET=math

# model config
MODEL=r1_1.5b
# MODEL=r1_7b
# MODEL=qwen_math_7b

# OOD
OOD=True

# PPO
PPO=False

# hyperparameter
LAMBDA_C=5
LAMBDA_W_E=5
LAMBDA_W_N=2
I_THRESHOLD=0.5
SLOPE_START_EPOCH=50
SLOPE_PERIOD=10
SLOPE_THRESHOLD=2.5


PROJECT_NAME='luck'
EXP_NAME=${MODEL}_${DATASET}_c${LAMBDA_C}_we${LAMBDA_W_E}_wn${LAMBDA_W_N}_i${I_THRESHOLD}_ss${SLOPE_START_EPOCH}_sp${SLOPE_PERIOD}_sth${SLOPE_THRESHOLD}_ppo_${PPO}
OUTPUT_DIR=/Bingo-submit/checkpoints/${PROJECT_NAME}/${EXP_NAME}


if [ $MODEL = 'r1_1.5b' ]; then
    INIT_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
elif [ $MODEL = 'r1_7b' ]; then
    INIT_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
elif [ $MODEL = 'qwen_math_7b' ]; then
    INIT_MODEL=Qwen/Qwen2.5-Math-7B-Instruct
fi


# training config
if [ $MODEL = 'r1_1.5b' ]; then
    NODE_NUM=1
    GPU_PER_NODE_NUM=2
    MINI_BATCH_SIZE=256
    MICRO_BATCH_SIZE_PER_GPU=4
    GPU_MEMORY_UTILIZATION=0.3
elif [ $MODEL = 'r1_7b' ]; then
    NODE_NUM=1
    GPU_PER_NODE_NUM=4
    MINI_BATCH_SIZE=256
    MICRO_BATCH_SIZE_PER_GPU=1
    GPU_MEMORY_UTILIZATION=0.3
elif [ $MODEL = 'qwen_math_7b' ]; then
    NODE_NUM=1
    GPU_PER_NODE_NUM=4
    MINI_BATCH_SIZE=256
    MICRO_BATCH_SIZE_PER_GPU=8
    GPU_MEMORY_UTILIZATION=0.2
fi


train_files=/Bingo-submit/dataset/${DATASET}/data/train.parquet
test_files=/Bingo-submit/dataset/${DATASET}/data/test.parquet

math500_test_files=/Bingo-submit/dataset/math500/data/test.parquet
gsm8k_test_files=/Bingo-submit/dataset/gsm8k/data/test.parquet
aime_test_files=/Bingo-submit/dataset/aime/data/test.parquet
theoremqa_test_files=/Bingo-submit/dataset/theoremqa/data/test.parquet

if [ $OOD = "True" ]; then
    # test_files="['$test_files', '$math500_test_files', '$gsm8k_test_files', '$aime_test_files', '$theoremqa_test_files']"
    test_files="['$math500_test_files', '$gsm8k_test_files', '$aime_test_files', '$theoremqa_test_files']"
fi

python3 -m verl.trainer.main_ppo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$INIT_MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$GPU_PER_NODE_NUM \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.response_length=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=$INIT_MODEL \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.lambda_c=$LAMBDA_C \
    algorithm.lambda_w_n=$LAMBDA_W_N \
    algorithm.lambda_w_e=$LAMBDA_W_E \
    algorithm.i_threshold=$I_THRESHOLD \
    algorithm.slope_start_epoch=$SLOPE_START_EPOCH \
    algorithm.slope_period=$SLOPE_PERIOD \
    algorithm.slope_threshold=$SLOPE_THRESHOLD \
    algorithm.gamma=1.0 \
    algorithm.adv_estimator=lgae \
    algorithm.ppo=$PPO \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXP_NAME \
    trainer.n_gpus_per_node=$GPU_PER_NODE_NUM \
    trainer.nnodes=$NODE_NUM \
    trainer.save_freq=20 \
    trainer.test_freq=10 \
    +trainer.val_before_train=True \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.total_epochs=50 $@
