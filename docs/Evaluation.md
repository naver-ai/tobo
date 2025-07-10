
## Vision-based Robot Policy Learning
### Franka Kitchen
We modify [R3M](https://github.com/facebookresearch/r3m) for VisionTransformer backbones and evaluate on five tasks for four seeds
* tasks
    * kitchen_knob1_on-v3
    * kitchen_light_on-v3
    * kitchen_sdoor_open-v3
    * kitchen_ldoor_open-v3
    * kitchen_micro_open-v3
```
python hydra_launcher.py  hydra/launcher=local hydra/output=local env=${task} camera=${camera} pixel_based=true embedding=vit num_demos=25 env_kwargs.load_path=${model} bc_kwargs.finetune=false proprio=9 job_name=try_${seed} seed=${seed}
```

### CortexBench
We use the evaluation code in [eai-vc](https://github.com/facebookresearch/eai-vc)

### RLBench
We modify [ARM](https://github.com/stepjam/ARM) for evaluation on RLBench


## Video Label Propagation
We use the evaluation code in [CropMAE](https://github.com/alexandre-eymael/CropMAE/)
