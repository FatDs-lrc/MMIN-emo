set -e
for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_deep --model=mapping_test --gpu_ids=2
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--acoustic_ft_type=A --lexical_ft_type=L --visual_ft_type=V
--fusion_size=128 --mid_layers_fusion=256,128
--mapping_layers=256,256,128
--output_dim=4 --modality=AVL
--niter=30 --niter_decay=70 --verbose
--batch_size=256 --lr=1e-3 --dropout_rate=0.5 --run_idx=2
--name=map_deep_feature --suffix=run{run_idx}
--cvNo=$i"

# --name=ef_A --suffix=Adnn{mid_layers_a}_Vlstm{hidden_size}_{embd_size_v}_Lcnn{embd_size_l}_fusion{mid_layers_fusion}run{run_idx}

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done