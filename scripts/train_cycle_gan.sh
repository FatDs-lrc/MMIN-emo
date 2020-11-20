set -e
for i in `seq 1 1 10`;
do

cmd="python train_gan.py --dataset_mode=iemocap_deep --model=cycle_gan_deep --gpu_ids=0
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=100
--acoustic_ft_type=A --lexical_ft_type=L --visual_ft_type=V
--G_layers=128,100,72,100,128 --D_layers=128,64,1 --gan_mode=lsgan
--niter=60 --niter_decay=90 --verbose
--batch_size=1 --lr=1e-3  --run_idx=1
--convert_type=L2A
--name=cycle_gan --suffix={convert_type}_run{run_idx}
--cvNo=$i"

# --name=ef_A --suffix=Adnn{mid_layers_a}_Vlstm{hidden_size}_{embd_size_v}_Lcnn{embd_size_l}_fusion{mid_layers_fusion}run{run_idx}

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done