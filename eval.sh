DATASET_ROOT="./dataset/dtu-ours-hires-test-1200/"

# Task name
TASK_NAME="np-cvp-mvsnet"
CKPT_NAME="model_000001.ckpt"

# Checkpoint
LOAD_CKPT_DIR="./ckpts/$TASK_NAME/$CKPT_NAME"

# Output dir
OUT_DIR="./out/${TASK_NAME}_${CKPT_NAME}/"

CUDA_VISIBLE_DEVICES=0 python3 -m pdb -c continue eval.py \
\
--info=$TASK_NAME \
--mode="test" \
\
--dataset="dtu" \
--dataset_root=$DATASET_ROOT \
--imgsize=1200 \
--depth_h=1152 \
--depth_w=1600 \
--vselection="mvsnet" \
--nsrc=10 \
--nbadsrc=0 \
--nscale=4 \
--gtdepth=1 \
--eval_precision=16 \
--feature_ch 8 16 32 64 \
--gwc_groups 2 4 4 8 \
--target_d 8 16 32 48 \
\
--init_search_mode='uniform' \
--costmetric='gwc_weighted_sum' \
\
--batch_size=1 \
\
--loadckpt=$LOAD_CKPT_DIR \
--logckptdir=$CKPT_DIR \
\
--outdir=$OUT_DIR