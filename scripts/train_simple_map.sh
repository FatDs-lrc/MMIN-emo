set -e
for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_deep --model=simple_mapping --gpu_ids=3
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=2
--acoustic_ft_type=A --lexical_ft_type=L --visual_ft_type=V
--mapping_layers=128,100,72
--niter=20 --niter_decay=30 --verbose
--batch_size=256 --lr=1e-3 --dropout_rate=0.5 --run_idx=2
--mse_weight=0.1 --cycle_weight=0.1
--model_type=L2A
--name=simple_map_ef_feat_input_latent --suffix={model_type}_mse{mse_weight}_cycle{cycle_weight}_run{run_idx}
--cvNo=$i"

# --name=ef_A --suffix=Adnn{mid_layers_a}_Vlstm{hidden_size}_{embd_size_v}_Lcnn{embd_size_l}_fusion{mid_layers_fusion}run{run_idx}

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done