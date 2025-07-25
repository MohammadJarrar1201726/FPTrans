# %load kaggle/working/FPTrans/networks/FPTrans.py
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D
from torch.hub import download_url_to_file

from constants import pretrained_weights, model_urls
from core.losses import get as get_loss
from networks import vit
from utils_.misc import interpb, interpn


class Residual(nn.Module):
    def __init__(self, layers, up=2):
        super().__init__()
        self.layers = layers
        self.up = up

    def forward(self, x):
        h, w = x.shape[-2:]
        x_up = interpb(x, (h * self.up, w * self.up))
        x = x_up + self.layers(x)
        return x


class FPTrans(nn.Module):
    def __init__(self, opt, logger):
        super(FPTrans, self).__init__()
        self.opt = opt
        self.logger = logger
        self.shot = opt.shot
        self.drop_dim = opt.drop_dim
        self.drop_rate = opt.drop_rate
        self.drop2d_kwargs = {'drop_prob': opt.drop_rate, 'block_size': opt.block_size}

        # Check existence.
        pretrained = self.get_or_download_pretrained(opt.backbone, opt.tqdm)

        # Main model
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', vit.vit_model(opt.backbone,
                                       opt.height,
                                       pretrained=pretrained,
                                       num_classes=0,
                                       opt=opt,
                                       logger=logger))
        ]))
        embed_dim = vit.vit_factory[opt.backbone]['embed_dim']
        self.purifier = self.build_upsampler(embed_dim)
        self.__class__.__name__ = f"FPTrans/{opt.backbone}"

        # Pretrained model
        self.original_encoder = vit.vit_model(opt.backbone,
                                              opt.height,
                                              pretrained=pretrained,
                                              num_classes=0,
                                              opt=opt,
                                              logger=logger,
                                              original=True)
        for var in self.original_encoder.parameters():
            var.requires_grad = False

        # Define pair-wise loss
        self.pairwise_loss = get_loss(opt, logger, loss='pairwise')
        # Background sampler
        self.bg_sampler = np.random.RandomState(1289)

        logger.info(' ' * 5 + f"==> Model {self.__class__.__name__} created")

    def build_upsampler(self, embed_dim):
        return Residual(nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(self.drop_rate) if self.drop_dim == 1 else DropBlock2D(**self.drop2d_kwargs),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(self.drop_rate) if self.drop_dim == 1 else DropBlock2D(**self.drop2d_kwargs),
            nn.Conv2d(256, embed_dim, kernel_size=1),
        ))

    def forward(self, x, s_x, s_y, y=None, out_shape=None):
        """

        Parameters
        ----------
        x: torch.Tensor
            [B, C, H, W], query image
        s_x: torch.Tensor
            [B, S, C, H, W], support image
        s_y: torch.Tensor
            [B, S, H, W], support mask
        y: torch.Tensor
            [B, 1, H, W], query mask, used for calculating the pair-wise loss
        out_shape: list
            The shape of the output predictions. If not provided, it is default
            to the last two dimensions of `y`. If `y` is also not provided, it is
            default to the [opt.height, opt.width].

        Returns
        -------
        output: dict
            'out': torch.Tensor
                logits that predicted by feature proxies
            'out_prompt': torch.Tensor
                logits that predicted by prompt proxies
            'loss_pair': float
                pair-wise loss
        """
        B, S, C, H, W = s_x.size()
        img_cat = torch.cat((s_x, x.view(B, 1, C, H, W)), dim=1).view(B*(S+1), C, H, W)

        # Calculate class-aware prompts
        with torch.no_grad():
            inp = s_x.view(B * S, C, H, W)
            # Forward
            sup_feat = self.original_encoder(inp)['out']
            _, c, h0, w0 = sup_feat.shape
            sup_mask = interpn(s_y.view(B*S, 1, H, W), (h0, w0))                                # [BS, 1, h0, w0]
            sup_mask_fg = (sup_mask == 1).float()                                               # [BS, 1, h0, w0]
            # Calculate fg and bg tokens
            fg_token = (sup_feat * sup_mask_fg).sum((2, 3)) / (sup_mask_fg.sum((2, 3)) + 1e-6)
            fg_token = fg_token.view(B, S, c).mean(1, keepdim=True)  # [B, 1, c]
            bg_token = self.compute_multiple_prototypes(
                self.opt.bg_num,
                sup_feat.view(B, S, c, h0, w0),
                sup_mask == 0,
                self.bg_sampler
            ).transpose(1, 2)    # [B, k, c]

        # Forward
        img_cat = (img_cat, (fg_token, bg_token))
        backbone_out = self.encoder(img_cat)

        features = self.purifier(backbone_out['out'])               # [B(S+1), c, h, w]
        _, c, h, w = features.size()
        features = features.view(B, S+1, c, h, w)                   # [B, S+1, c, h, w]
        sup_fts, qry_fts = features.split([S, 1], dim=1)            # [B, S, c, h, w] / [B, 1, c, h, w]
        sup_mask = interpn(s_y.view(B * S, 1, H, W), (h, w))        # [BS, 1, h, w]

        pred = self.classifier(sup_fts, qry_fts, sup_mask)          # [B, 2, h, w]

        # Output
        if not out_shape:
            out_shape = y.shape[-2:] if y is not None else (H, W)
        out = interpb(pred, out_shape)    # [BQ, 2, *, *]
        output = dict(out=out)

        if self.training and y is not None:
            # Pairwise loss
            x1 = sup_fts.flatten(3)                 # [B, S, C, N]
            y1 = sup_mask.view(B, S, -1).long()     # [B, S, N]
            x2 = qry_fts.flatten(3)                 # [B, 1, C, N]
            y2 = interpn(y.float(), (h, w)).flatten(2).long()   # [B, 1, N]
            output['loss_pair'] = self.pairwise_loss(x1, y1, x2, y2)

            # Prompt-Proxy prediction
            fg_token = self.purifier(backbone_out['tokens']['fg'])[:, :, 0, 0]        # [B, c]
            bg_token = self.purifier(backbone_out['tokens']['bg'])[:, :, 0, 0]        # [B, c]
            bg_token = bg_token.view(B, self.opt.bg_num, c).transpose(1, 2)     # [B, c, k]
            pred_prompt = self.compute_similarity(fg_token, bg_token, qry_fts.reshape(-1, c, h, w))

            # Up-sampling
            pred_prompt = interpb(pred_prompt, (H, W))
            output['out_prompt'] = pred_prompt

        return output

    def classifier(self, sup_fts, qry_fts, sup_mask):
        """

        Parameters
        ----------
        sup_fts: torch.Tensor
            [B, S, c, h, w]
        qry_fts: torch.Tensor
            [B, 1, c, h, w]
        sup_mask: torch.Tensor
            [BS, 1, h, w]

        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w]

        """
        B, S, c, h, w = sup_fts.shape

        # FG proxies
        sup_fg = (sup_mask == 1).view(-1, 1, h * w)  # [BS, 1, hw]
        fg_vecs = torch.sum(sup_fts.reshape(-1, c, h * w) * sup_fg, dim=-1) / (sup_fg.sum(dim=-1) + 1e-5)     # [BS, c]
        # Merge multiple shots
        fg_proto = fg_vecs.view(B, S, c).mean(dim=1)    # [B, c]

        # BG proxies
        bg_proto = self.compute_multiple_prototypes(self.opt.bg_num, sup_fts, sup_mask == 0, self.bg_sampler)

        # Calculate cosine similarity
        qry_fts = qry_fts.reshape(-1, c, h, w)
        pred = self.compute_similarity(fg_proto, bg_proto, qry_fts)   # [B, 2, h, w]
        return pred

    @staticmethod
    def compute_multiple_prototypes(bg_num, sup_fts, sup_bg, sampler):
        """

        Parameters
        ----------
        bg_num: int
            Background partition numbers
        sup_fts: torch.Tensor
            [B, S, c, h, w], float32
        sup_bg: torch.Tensor
            [BS, 1, h, w], bool
        sampler: np.random.RandomState

        Returns
        -------
        bg_proto: torch.Tensor
            [B, c, k], where k is the number of background proxies

        """
        B, S, c, h, w = sup_fts.shape
        bg_mask = sup_bg.view(B, S, h, w)    # [B, S, h, w]
        batch_bg_protos = []

        for b in range(B):
            bg_protos = []
            for s in range(S):
                bg_mask_i = bg_mask[b, s]     # [h, w]

                # Check if zero
                with torch.no_grad():
                    if bg_mask_i.sum() < bg_num:
                        bg_mask_i = bg_mask[b, s].clone()    # don't change original mask
                        bg_mask_i.view(-1)[:bg_num] = True

                # Iteratively select farthest points as centers of background local regions
                all_centers = []
                first = True
                pts = torch.stack(torch.where(bg_mask_i), dim=1)     # [N, 2]
                for _ in range(bg_num):
                    if first:
                        i = sampler.choice(pts.shape[0])
                        first = False
                    else:
                        dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                        # choose the farthest point
                        i = torch.argmax((dist ** 2).sum(-1).min(1)[0])
                    pt = pts[i]   # center y, x
                    all_centers.append(pt)
            
                # Assign bg labels for bg pixels
                dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                bg_labels = torch.argmin((dist ** 2).sum(-1), dim=1)

                # Compute bg prototypes
                bg_feats = sup_fts[b, s].permute(1, 2, 0)[bg_mask_i]    # [N, c]
                for i in range(bg_num):
                    proto = bg_feats[bg_labels == i].mean(0)    # [c]
                    bg_protos.append(proto)

            bg_protos = torch.stack(bg_protos, dim=1)   # [c, k]
            batch_bg_protos.append(bg_protos)
        bg_proto = torch.stack(batch_bg_protos, dim=0)  # [B, c, k]
        return bg_proto

    @staticmethod
    def compute_similarity(fg_proto, bg_proto, qry_fts, dist_scalar=20):
        """
        Parameters
        ----------
        fg_proto: torch.Tensor
            [B, c], foreground prototype
        bg_proto: torch.Tensor
            [B, c, k], multiple background prototypes
        qry_fts: torch.Tensor
            [B, c, h, w], query features
        dist_scalar: int
            scale factor on the results of cosine similarity

        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w], predictions
        """
        fg_distance = F.cosine_similarity(
            qry_fts, fg_proto[..., None, None], dim=1) * dist_scalar        # [B, h, w]
        if len(bg_proto.shape) == 3:    # multiple background protos
            bg_distances = []
            for i in range(bg_proto.shape[-1]):
                bg_p = bg_proto[:, :, i]
                bg_d = F.cosine_similarity(
                    qry_fts, bg_p[..., None, None], dim=1) * dist_scalar        # [B, h, w]
                bg_distances.append(bg_d)
            bg_distance = torch.stack(bg_distances, dim=0).max(0)[0]
        else:   # single background proto
            bg_distance = F.cosine_similarity(
                qry_fts, bg_proto[..., None, None], dim=1) * dist_scalar        # [B, h, w]
        pred = torch.stack((bg_distance, fg_distance), dim=1)               # [B, 2, h, w]

        return pred

    # def load_weights(self, ckpt_path, logger, strict=True):
    #     """

    #     Parameters
    #     ----------
    #     ckpt_path: Path
    #         path to the checkpoint
    #     logger
    #     strict: bool
    #         strict mode or not

    #     """
    #     weights = torch.load(str(ckpt_path), map_location='cpu' , weights_only=False)
    #     if "model_state" in weights:
    #         weights = weights["model_state"]
    #     if "state_dict" in weights:
    #         weights = weights["state_dict"]
    #     weights = {k.replace("module.", ""): v for k, v in weights.items()}
    #     # Update with original_encoder
    #     weights.update({k: v for k, v in self.state_dict().items() if 'original_encoder' in k})

    #     self.load_state_dict(weights, strict=strict)        
    #     logger.info(' ' * 5 + f"==> Model {self.__class__.__name__} initialized from {ckpt_path}")

    # def load_weights(self, ckpt_path, logger, strict=False):
    #   """
    #   Load weights from a checkpoint, handling unexpected keys and size mismatches.

    #   Parameters
    #   ----------
    #   ckpt_path: Path
    #       Path to the checkpoint file.
    #   logger:
    #       Logger instance for logging messages.
    #   strict: bool
    #       Whether to enforce strict loading (default: False).
    #   """
    #   # Load checkpoint weights
    #   weights = torch.load(str(ckpt_path), map_location='cpu' , weights_only=False)
    #   if "model_state" in weights:
    #       weights = weights["model_state"]
    #   if "state_dict" in weights:
    #       weights = weights["state_dict"]
    #   weights = {k.replace("module.", ""): v for k, v in weights.items()}

    #   # Get model’s state_dict for comparison
    #   model_state_dict = self.state_dict()

    #   # Filter weights to only include keys present in the model
    #   weights = {k: v for k, v in weights.items() if k in model_state_dict}

    #   # Handle size mismatches
    #   for key in weights:
    #       if weights[key].shape != model_state_dict[key].shape:
    #           if key == 'encoder.backbone.prompt_tokens':
    #               ckpt_shape = weights[key].shape  # [90, 12, 768]
    #               model_shape = model_state_dict[key].shape  # [360, 12, 768]
    #               if ckpt_shape[1:] == model_shape[1:]:  # Check if heads and embed_dim match
    #                   num_tokens_ckpt = ckpt_shape[0]
    #                   num_tokens_model = model_shape[0]
    #                   if num_tokens_model % num_tokens_ckpt == 0:
    #                       repeat_times = num_tokens_model // num_tokens_ckpt
    #                       weights[key] = weights[key].repeat(repeat_times, 1, 1)
    #                       logger.info(f"Adjusted prompt_tokens from {ckpt_shape} to {model_shape} by repeating.")
    #                   else:
    #                       logger.warning(f"Cannot adjust prompt_tokens: {ckpt_shape} to {model_shape} not evenly divisible.")
    #               else:
    #                   logger.warning(f"Shape mismatch for {key}: {ckpt_shape} vs {model_shape}")
    #           elif key == 'encoder.backbone.pos_embed':
    #               ckpt_shape = weights[key].shape  # [1, 902, 768]
    #               model_shape = model_state_dict[key].shape  # [1, 578, 768]
    #               if ckpt_shape[0] == model_shape[0] and ckpt_shape[2] == model_shape[2]:
    #                   # Separate class token, patch embeddings, and dist token
    #                   class_token = weights[key][:, :1, :]  # [1, 1, 768]
    #                   dist_token = weights[key][:, -1:, :]  # [1, 1, 768]
    #                   patch_embeds = weights[key][:, 1:-1, :]  # [1, 900, 768]
    #                   num_patches_ckpt = patch_embeds.shape[1]
    #                   num_patches_model = model_shape[1] - 2  # 576

    #                   # Compute grid sizes (assuming square patch grid)
    #                   grid_size_ckpt = int(round(num_patches_ckpt ** 0.5))  # e.g., 30
    #                   grid_size_model = int(round(num_patches_model ** 0.5))  # 24
    #                   if grid_size_ckpt ** 2 == num_patches_ckpt and grid_size_model ** 2 == num_patches_model:
    #                       # Reshape to [1, embed_dim, grid_size, grid_size] for interpolation
    #                       patch_embeds = patch_embeds.permute(0, 2, 1).reshape(1, -1, grid_size_ckpt, grid_size_ckpt)
    #                       # Interpolate to model grid size
    #                       patch_embeds = F.interpolate(patch_embeds, size=(grid_size_model, grid_size_model), 
    #                                                   mode='bilinear', align_corners=False)
    #                       # Reshape back to [1, num_patches_model, embed_dim]
    #                       patch_embeds = patch_embeds.reshape(1, -1, num_patches_model).permute(0, 2, 1)
    #                       # Reassemble pos_embed
    #                       weights[key] = torch.cat([class_token, patch_embeds, dist_token], dim=1)
    #                       logger.info(f"Interpolated pos_embed from {ckpt_shape} to {model_shape}.")
    #                   else:
    #                       logger.warning(f"Cannot interpolate pos_embed: non-square grid detected.")
    #               else:
    #                   logger.warning(f"Shape mismatch for {key}: {ckpt_shape} vs {model_shape}")
    #           else:
    #               logger.warning(f"Size mismatch for {key}: {weights[key].shape} vs {model_state_dict[key].shape}")

    #   # Load the adjusted weights
    #   self.load_state_dict(weights, strict=strict)
    #   logger.info(' ' * 5 + f"==> Model {self.__class__.__name__} initialized from {ckpt_path}")
    def load_weights(self, ckpt_path, logger, strict=False):
        import torch
        from networks import vit  # Ensure resize_pos_embed is available
    
        # Load the checkpoint
        weights = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if "model_state" in weights:
            weights = weights["model_state"]
        if "state_dict" in weights:
            weights = weights["state_dict"]
        weights = {k.replace("module.", ""): v for k, v in weights.items()}
    
        # Get the model's state dictionary
        model_state_dict = self.state_dict()
    
        # Resize pos_embed if necessary
        if 'encoder.backbone.pos_embed' in weights:
            pos_embed_ckpt = weights['encoder.backbone.pos_embed']
            pos_embed_model = model_state_dict['encoder.backbone.pos_embed']
            if pos_embed_ckpt.shape != pos_embed_model.shape:
                gs_new = self.encoder.backbone.patch_embed.grid_size  # e.g., (24, 24)
                if gs_new != (30, 30):
                    logger.warning(f"Expected gs_new=(30, 30) for 480x480, got {gs_new}")
                num_tokens_model = self.encoder.backbone.num_tokens  # e.g., 2
                # weights['encoder.backbone.pos_embed'] = vit.resize_pos_embed(
                #     pos_embed_ckpt, pos_embed_model, num_tokens_model, gs_new
                # )
                gs_new_h = gs_new_w = 480 // 16  # 30 for 480x480
                num_tokens = 1  # Standard DeiT (or 2 for distilled)
                print(f"Resizing pos_embed: input shape {ckpt['pos_embed'].shape}, gs_new_h={gs_new_h}, gs_new_w={gs_new_w}, num_tokens={num_tokens}")
                ckpt['pos_embed'] = resize_pos_embed(
                    ckpt['pos_embed'], gs_new_h=gs_new_h, gs_new_w=gs_new_w, num_tokens=num_tokens
                )
                print(f"Resized pos_embed shape: {ckpt['pos_embed'].shape}")
                
        # Filter weights to only include keys that match the model’s state_dict in both name and shape
        filtered_weights = {
            k: v for k, v in weights.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
    
        # Load the filtered weights with strict=False
        missing_keys, unexpected_keys = self.load_state_dict(filtered_weights, strict=False)
        if missing_keys:
            logger.info(f"Missing keys (ignored): {missing_keys}")
        if unexpected_keys:
            logger.info(f"Unexpected keys (ignored): {unexpected_keys}")
    
        logger.info(f"==> Model {self.__class__.__name__} initialized from {ckpt_path}")
    @staticmethod
    def get_or_download_pretrained(backbone, progress):
        if backbone not in pretrained_weights:
            raise ValueError(f'Not supported backbone {backbone}. '
                             f'Available backbones: {list(pretrained_weights.keys())}')

        cached_file = Path(pretrained_weights[backbone])
        if cached_file.exists():
            return cached_file

        # Try to download
        url = model_urls[backbone]
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, str(cached_file), progress=progress)
        return cached_file

    def get_params_list(self):
        params = []
        for var in self.parameters():
            if var.requires_grad:
                params.append(var)
        return [{'params': params}]

