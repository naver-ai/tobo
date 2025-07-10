# Train

We pre-train our models on Kinetics-400 for 400 epochs. 

```
accum_iter=8
aux_path=sm
batch_size=24
epochs=400
local_data_path=/mnt/tmp/kinetics400/videos
mask_ratio=0.9
max_distance=96
max_frames=1
mim_name=tobo
min_frames=1
model_name=vit_small_patch16
num_frames=2
repeated_sampling=2
save_path=/mnt/tmp/checkpoints
tgt_path=csm
w_tobo=1.0
w_mim=1.0
warmup_epochs=40
num_gpus_per_node=$(nvidia-smi -L | wc -l)

python -m torch.distributed.launch --nproc_per_node=${num_gpus_per_node} main_pretrain_semae.py \
     --batch_size ${batch_size} \
     --accum_iter ${accum_iter} \
     --model semae_${model_name} \
     --epochs ${epochs} \
     --warmup_epochs ${warmup_epochs} \
     --data_path ${local_data_path} \
     --output_dir ${save_path}/output \
     --norm_pix_loss \
     --repeated_sampling ${repeated_sampling} \
     --mask_ratio ${mask_ratio} \
     --max_frames ${max_frames} \
     --min_frames ${min_frames} \
     --num_frames ${num_frames} \
     --max_distance ${max_distance} \
     --tgt_path ${tgt_path} \
     --se_path ${aux_path} \
     --w_mim ${w_mim} \
     --w_tobo ${w_tobo}
```
