set -e
for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_missAL --model=cycle_mapping --gpu_ids=2
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=4
--acoustic_ft_type=IS10 --lexical_ft_type=text
--input_dim_a=1582 --mid_layers_a=512,256,128
--input_dim_l=1024 --embd_size_l=128
--fusion_size=256 --mid_layers_fusion=256,128 --output_dim=4 
--mapping_layers=128,128,128
--niter=20 --niter_decay=30 --verbose
--batch_size=256 --lr=1e-3 --dropout_rate=0.5 --run_idx=8
--name=AL_mapping --suffix=_run{run_idx}
--cvNo=$i"

# --name=ef_AVL --suffix=Adnn{mid_layers_a}_Vlstm{hidden_size}_{embd_method}_Lcnn{embd_size_l}_fusion{mid_layers_fusion}run{run_idx}

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
done