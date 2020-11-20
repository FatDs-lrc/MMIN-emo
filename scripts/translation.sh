set -e
for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_align --model=translation --gpu_ids=4
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--input_modality=L --aug_modality1=A 
--fusion_size=128 --mid_layers_fusion=128,128 --output_dim=4 
--niter=40 --niter_decay=60 --verbose --init_type=xavier
--lambda_ce=1.0 --lambda_mse1=0.1 --lambda_mse2=0.1 --lambda_cycle=0.1
--batch_size=128 --lr=1e-3 --dropout_rate=0.5 --run_idx=1
--name=translation --suffix={input_modality}_{aug_modality1}_{aug_modality2}_run{run_idx}
--cvNo=$i"

# --name=ef_A --suffix=Adnn{mid_layers_a}_Vlstm{hidden_size}_{embd_size_v}_Lcnn{embd_size_l}_fusion{mid_layers_fusion}run{run_idx}

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done