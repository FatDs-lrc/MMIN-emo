set -e
feat_type=$1
feat_modality=$2
run_idx=$3
gpu_ids=$4
for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_ablation --model=emotion_ablation --gpu_ids=$gpu_ids
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--feat_modality=$feat_modality --feat_type=$feat_type
--fusion_layers=128,128 --output_dim=4
--niter=20 --niter_decay=20 --verbose
--batch_size=256 --lr=1e-3 --dropout_rate=0.5 --run_idx=$run_idx
--name=emo_ablation --suffix={feat_type}_{feat_modality}_run{run_idx}
--cvNo=$i"

# --name=ef_A --suffix=Adnn{mid_layers_a}_Vlstm{hidden_size}_{embd_size_v}_Lcnn{embd_size_l}_fusion{mid_layers_fusion}run{run_idx}

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done