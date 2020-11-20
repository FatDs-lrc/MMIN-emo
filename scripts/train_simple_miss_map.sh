set -e
for i in `seq 1 1 10`;
do

cmd="python train_autoencoder.py --dataset_mode=iemocap_deep_miss --model=simple_miss_map --gpu_ids=2
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=2
--acoustic_ft_type=A --lexical_ft_type=L --visual_ft_type=V
--mapping_layers=256,192,128
--niter=60 --niter_decay=90 --verbose
--batch_size=256 --lr=1e-3 --dropout_rate=0.5 --run_idx=1
--name=simple_map --suffix=run{run_idx}
--cvNo=$i"

# --name=ef_A --suffix=Adnn{mid_layers_a}_Vlstm{hidden_size}_{embd_size_v}_Lcnn{embd_size_l}_fusion{mid_layers_fusion}run{run_idx}

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done