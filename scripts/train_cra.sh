set -e
for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_miss --model=CRA_simple --gpu_ids=4
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=4
--acoustic_ft_type=IS10 --lexical_ft_type=text --visual_ft_type=denseface
--input_dim_a=1582 --mid_layers_a=512,256,128
--input_dim_v=342 --hidden_size_v=128 --embd_size_v=128 --embd_method=maxpool
--input_dim_l=1024 --embd_size_l=128
--fusion_size=256 --mid_layers_fusion=256,128 --output_dim=4 
--ce_weight=1 --mse_weight=1 --cycle_weight=0.1
--AE_layers=256,128 --n_blocks=5
--teacher_path='checkpoints/multi_fusion_manyCE_run1'
--niter=100 --niter_decay=100 --verbose --miss_data_iters=4
--batch_size=256 --lr=1e-3 --dropout_rate=0.5 --run_idx=2
--miss2_rate=0.5 --miss_num=mix
--name=cra_simple_supervision_AVL --suffix=AE{AE_layers}_run{run_idx}
--cvNo=$i"

# miss2_fix{miss2_rate}_real{real_data_rate}_AE{AE_layers}_blocks{n_blocks}_ce{ce_weight}_mse_{mse_weight}

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
done