set -e
for i in `seq 1 1 10`;
# for i in 1 4 5 6 7 8 9 10;
# for i in 1 2 3 4 7 8 9;
# for i in 1;
do

cmd="python train_miss_mix.py --dataset_mode=iemocap_miss --model=new_simpleAE --gpu_ids=7
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=4
--acoustic_ft_type=IS10 --lexical_ft_type=text --visual_ft_type=denseface
--input_dim_a=1582 --mid_layers_a=512,256,128
--input_dim_v=342 --hidden_size_v=128 --embd_size_v=128 --embd_method=maxpool
--input_dim_l=1024 --embd_size_l=128
--fusion_size=256 --mid_layers_fusion=256,128 --output_dim=4 
--ce_weight=1 --mse_weight=0.2 --cycle_weight=0.2
--AE_layers=256,128,64 --n_blocks=1
--teacher_path='checkpoints/ef_AVL_Adnn512,256,128_Vlstm128_maxpool_Lcnn128_fusion256,128run1/'
--niter=40 --niter_decay=40 --verbose
--batch_size=256 --lr=1e-3 --dropout_rate=0.5 --run_idx=1
--miss2_rate=0.5 --miss_num=mix
--name=CRA_block_ablation_simple_AE --suffix=AE{AE_layers}_blocks{n_blocks}_run{run_idx}
--cvNo=$i"
# 'checkpoints/multi_fusion_manyCE_run2/'
# --teacher_path='checkpoints/ef_AVL_Adnn512,256,128_Vlstm128_maxpool_Lcnn128_fusion256,128run1/'
# miss2_fix{miss2_rate}_real{real_data_rate}_AE{AE_layers}_blocks{n_blocks}_ce{ce_weight}_mse_{mse_weight}

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
done