set -e
for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_deep --model=residual_ae --gpu_ids=0
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=4
--input_dim_a=128 --input_dim_v=128 --input_dim_l=128 
--AE_layers=128,64,32 --n_blocks=5
--teacher_path='checkpoints/ef_AVL_Adnn512,256,128_Vlstm128_maxpool_Lcnn128_fusion256,128run1'
--output_dim=4 --dropout_rate=0.5
--niter=40 --niter_decay=60
--batch_size=256 --lr=1e-3 --run_idx=2
--name=RAE --suffix=layers{AE_layers}_blocks{n_blocks}_bs{batch_size}_dp{dropout_rate}_run{run_idx}
--cvNo=$i"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done