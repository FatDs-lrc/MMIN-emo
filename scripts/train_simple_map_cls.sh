set -e
for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_deep --model=simple_map_cls --gpu_ids=1
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=2
--acoustic_ft_type=A --lexical_ft_type=L --visual_ft_type=V
--niter=30 --niter_decay=70 --verbose
--model_type=V --run_idx=2
--name=simple_one_modality --suffix={model_type}_run{run_idx}
--cvNo=$i"

# --name=ef_A --suffix=Adnn{mid_layers_a}_Vlstm{hidden_size}_{embd_size_v}_Lcnn{embd_size_l}_fusion{mid_layers_fusion}run{run_idx}

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done