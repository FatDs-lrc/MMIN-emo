set -e
data_type=$1
i=2

cmd="python train_baseline.py --dataset_mode=iemocap_analysis_cm 
--model=dnn --gpu_ids=4
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--data_type=$data_type --output_dim=6 
--niter=50 --niter_decay=50 --verbose
--batch_size=256 --lr=1e-3 --dropout_rate=0.5 --run_idx=1
--name=analysis --suffix={data_type}_run{run_idx}
--cvNo=$i"

# --name=ef_A --suffix=Adnn{mid_layers_a}_Vlstm{hidden_size}_{embd_size_v}_Lcnn{embd_size_l}_fusion{mid_layers_fusion}run{run_idx}

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
