from functools import partial
import random

import torch
import torch.nn as nn
import torch.distributions as td

from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import CrossAttention, Attention, DropPath, Mlp

from util.pos_embed import get_2d_sincos_pos_embed, get_sinusoid_encoding_table

class CrossAttention_tobo(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        num_frames=2,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.num_frames = num_frames

    def forward(self, x, kvx, num_frames=1, src_mask=None):
        B, N, C = x.shape
        _, kvN, _ = kvx.shape
        if src_mask != None:
            kv = (
                self.kv(kvx)
                .reshape(B // num_frames, kvN, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(kvx)
                .reshape(B, kvN, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        q = (
            self.q(x)
            .reshape(B, N, 1, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = (
            q[0],
            kv[0],
            kv[1],
        )  # make torchscript happy (cannot use tensor as tuple)

        if src_mask != None:
            k = k.repeat(num_frames, 1, 1, 1)
            v = v.repeat(num_frames, 1, 1, 1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if src_mask != None:
            src_mask = src_mask.unsqueeze(1)
            src_mask = src_mask.unsqueeze(1)
            src_mask = src_mask.repeat(1, self.num_heads, N, 1)
            attn = attn.masked_fill(src_mask == 0, -1e4)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CSABlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        num_frames=2,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm_kv1 = norm_layer(dim)
        self.cattn = CrossAttention_tobo(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            num_frames=num_frames,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm3 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm4 = norm_layer(dim)
        self.mlp2 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, kvx, num_frames=1, src_mask=None, path="csm"):
        if path=="csm":
            x = x + self.drop_path(
                self.cattn(self.norm1(x), self.norm_kv1(kvx), src_mask=src_mask, num_frames=num_frames)
            )
            x = x + self.drop_path(self.attn(self.norm3(x)))
        elif path=="sm":
            x = x + self.drop_path(self.attn(self.norm3(x)))
        elif path=="scm":
            x = x + self.drop_path(self.attn(self.norm3(x)))
            x = x + self.drop_path(
                self.cattn(self.norm1(x), self.norm_kv1(kvx), src_mask=src_mask, num_frames=num_frames)
            )
        x = x + self.mlp2(self.norm4(x))
        return x


class tobo(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=True,
        mask_ratio=0.75,
        mask_ratio_src=0.,
        batch_size=None,
        repeated_sampling=2,
        num_frames=None,
        min_frames=1,
        max_frames=3,
        tgt_path=None,
        tobo_path=None,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_embed_mae = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList(
            [
                CSABlock(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    num_frames = num_frames,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.mask_ratio = mask_ratio

        self.batch_size = batch_size * repeated_sampling
        self.num_frames = num_frames
        self.mask_ratio_src = mask_ratio_src
        self.tgt_path = tgt_path
        self.tobo_path = tobo_path

        self.min_frames = min_frames
        self.max_frames = max_frames
        self.initialize_weights()


    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches**0.5),
            cls_token=True,
        )
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0)
        )

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum("nchpwq->nhwpqc", x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder_asym(self, imgs, mask_ratio=0.0, num_frames=1, src=False):
        x = self.patch_embed(imgs)
        x = x + self.pos_embed[:, 1:, :]

        if mask_ratio != 0.0:
            x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        else:
            mask, ids_restore, ids_keep = None, None, None

        if src:
            x = x.reshape(self.batch_size, x.shape[1] * num_frames, -1)


        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore, ids_keep

    def forward_decoder_tobo(self, h, x, ids_restore, ids_keep, num_frames=1):
        h = self.decoder_embed_mae(h)  # remove cls tokens
        _, N, D = h.shape

        decoder_pos_embed = self.decoder_pos_embed.repeat(self.batch_size * num_frames, 1, 1)
        if num_frames == 1:
            pe_src = decoder_pos_embed[:,1:]
        else:
            pe_src = torch.gather(decoder_pos_embed[:, 1:], dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, h.shape[2]))
        h_cls = h[:, :1]
        h_ = h[:, 1:]
        kvx_h = h_ + pe_src.reshape(self.batch_size, h_.shape[1], h_.shape[2])
        kvx_h = torch.cat([h_cls + self.decoder_pos_embed[:, :1], kvx_h], dim=1)


        x = self.decoder_embed_mae(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x_ = x_ + self.decoder_pos_embed[:, 1:]
        x_cls = x[:, :1, :] + self.decoder_pos_embed[:, :1]
        x_h_cls = h_cls + self.decoder_pos_embed[:, :1]
        x1 = torch.cat([x_cls, x_], dim=1)  # append cls token
        x2 = torch.cat([x_h_cls, x_], dim=1)

        for blk in self.decoder_blocks:
            x1 = blk(x1, kvx=kvx_h, num_frames=num_frames, path=self.tgt_path)
            x2 = blk(x2, kvx=None, num_frames=num_frames, path=self.tobo_path)
        x = torch.cat([x1, x2], dim=0)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask=None):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        recon_loss = (pred - target) ** 2
        if mask is not None:
            recon_loss = recon_loss.mean(dim=-1)  # [N, L], mean loss per patch
            recon_loss = (recon_loss * mask).sum() / mask.sum()  # mean loss on removed patches
        else:
            recon_loss = recon_loss.mean()

        return recon_loss

    def forward(self, list_imgs, epoch, list_idx=None):
        B, C, H, W = list_imgs[0].shape

        src_imgs = list_imgs[0]
        tgt_imgs = list_imgs[-1]
        mask_ratio_src = 0.0
        num_sampled_imgs = 1


        src_h, src_mask, src_ids_restore, src_ids_keep = self.forward_encoder_asym(src_imgs, mask_ratio=mask_ratio_src, num_frames=num_sampled_imgs, src=True)
        tgt_h, tgt_mask, tgt_ids_restore, tgt_ids_keep = self.forward_encoder_asym(tgt_imgs, mask_ratio=self.mask_ratio)

        pred_masked = self.forward_decoder_tobo(src_h, tgt_h, tgt_ids_restore, src_ids_keep, num_frames=num_sampled_imgs)
        pred_masked_patch, pred_masked_tobo = pred_masked.chunk(2, dim=0)

        mae_loss = self.forward_loss(tgt_imgs, pred_masked_patch, tgt_mask)
        mae_loss_tobo = self.forward_loss(tgt_imgs, pred_masked_tobo, tgt_mask)


        return mae_loss, mae_loss_tobo




def tobo_vit_small_patch16_dec512d8b(**kwargs):
    model = tobo(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def tobo_vit_base_patch16_dec512d8b(**kwargs):
    model = tobo(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

def tobo_vit_large_patch16_dec512d8b(**kwargs):
    model = tobo(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

tobo_vit_small_patch16 = tobo_vit_small_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
tobo_vit_base_patch16 = tobo_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
tobo_vit_large_patch16 = tobo_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
