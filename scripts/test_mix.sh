set -e

cmd="python test_mix.py --dataset_mode=iemocap_miss --log_dir=./logs --simple
--checkpoints_dir=./checkpoints  --model=early_fusion_multi --method=mean
--name=new_cra_AE256,128,64_blocks5_run1"
# TS_acoustic_Spec300.0_Orth0.002_Center0.05_KDtemp4.0_layers512,256,128

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

