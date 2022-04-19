set -x; set -e;

TASK_NAME="np-cvp-mvsnet"
DTU_HIGHRES_TRAIN_PATH="dataset/dtu-ours-hires-train-512/"

CKPT_DIR="./ckpts/"
mkdir -p $CKPT_DIR

python3 -m torch.distributed.launch --nnodes 1 --node_rank 0 --nproc_per_node 1 ddp_amp_train.py \
\
--info="${TASK_NAME}" \
--mode="train" \
\
--dataset=dtu \
--dataset_root=$DTU_HIGHRES_TRAIN_PATH \
--imgsize=512 \
--vselection="mvsnet" \
--nsrc=2 \
--nbadsrc=2 \
--nscale=4 \
--gtdepth=1 \
--feature_ch 8 16 32 64 \
--gwc_groups 2 4 4 8 \
--target_d 8 16 32 48 \
\
--init_search_mode='uniform' \
--costmetric='gwc_weighted_sum' \
\
--epochs=27 \
--lr=0.001 \
--lrepochs="10,12,14,20:2" \
--wd=0.0 \
--batch_size=1 \
--summary_freq=1 \
--save_freq=1 \
--seed=1 \
--loss_function='BCE' \
--activate_level_itr 0 0 0 0 \
--final_edge_mask=1 \
--final_weight=0.1 \
--final_continue=1 \
\
--logckptdir=$CKPT_DIR