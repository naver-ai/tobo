import os

import torch

from util.pos_embed import interpolate_pos_embed


def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    print(checkpoint_key)
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    print(state_dict.keys())
    if checkpoint_key is not None and checkpoint_key in state_dict:
        print(f"Take key {checkpoint_key} in provided checkpoint dict")
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    state_dict = {k.replace("base_encoder.", ""): v for k, v in state_dict.items()}
    interpolate_pos_embed(model, state_dict)
    msg = model.load_state_dict(state_dict, strict=False)
    print(
        "Pretrained weights found at {} and loaded with msg: {}".format(
            pretrained_weights, msg
        )
    )
