set -e
for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_10fold --model=seqAE --gpu_ids=2
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=4
--acoustic_ft_type=IS10 --lexical_ft_type=text --visual_ft_type=denseface
--input_modality=V --output_modality=A --embd_method=attention --output_dim=4 
--A_layers=512,256
--ce_weight=1 --mse_weight=1 --cycle_weight=0.1
--niter=30 --niter_decay=20 --verbose
--batch_size=128 --lr=1e-3 --dropout_rate=0.5 --run_idx=1
--name=SeqAE_latent_utt_attention --suffix={input_modality}2{output_modality}_run{run_idx}
--cvNo=$i"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
done