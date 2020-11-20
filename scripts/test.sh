set -e

cmd="python test.py --dataset_mode=iemocap_deep --log_dir=./logs --simple
--checkpoints_dir=./checkpoints  --model=early_fusion_multi --method=mean
--name=new_cra_AE256,128,64_blocks5_run3"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

