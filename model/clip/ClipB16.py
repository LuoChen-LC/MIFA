import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from .Base import BaseFeatureExtractor


class ClipB16FeatureExtractor(BaseFeatureExtractor):
    def __init__(self):
        super(ClipB16FeatureExtractor, self).__init__()
        
        self.model = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-base-patch16",  
            output_hidden_states=True  
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.normalizer = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Lambda(lambda img: torch.clamp(img, 0.0, 255.0) / 255.0),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def forward(self, x):
        x_norm = self.normalizer(x)
        vision_outputs = self.model.vision_model(
            pixel_values=x_norm, 
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = vision_outputs.hidden_states 
        pooled_output = vision_outputs.pooler_output
        global_features = self.model.visual_projection(pooled_output)
        global_features = F.normalize(global_features, p=2, dim=-1)
        # ======================================================

        return global_features, hidden_states
    
class FineGrainedCLIPLoss(nn.Module):
    def __init__(self, 
                 use_global=True, 
                 use_intermediate=True, 
                 use_spatial_mask=True, 
                 target_layers=[3, 6, 9], 
                 lambda_fg=5.0):
        super(FineGrainedCLIPLoss, self).__init__()
        self.use_global = use_global
        self.use_intermediate = use_intermediate
        self.use_spatial_mask = use_spatial_mask
        self.target_layers = target_layers
        self.lambda_fg = lambda_fg

        if self.use_spatial_mask:
            self.register_buffer("spatial_mask", self._create_prior_face_mask())

    def _create_pixel_face_mask(self, H=224, W=224):
        mask = torch.zeros((H, W), dtype=torch.float32)

        h_start, h_end = int(H * 0.3), int(H * 0.7)
        w_start, w_end = int(W * 0.3), int(W * 0.7)
        mask[h_start:h_end, w_start:w_end] = 1.0

        return mask
    def _create_prior_face_mask(self, H=224, W=224, patch_size=16):
        pixel_mask = self._create_pixel_face_mask(H, W)  
        pixel_mask = pixel_mask.unsqueeze(0).unsqueeze(0)  

        pooled = F.avg_pool2d(
            pixel_mask,
            kernel_size=patch_size,
            stride=patch_size
        )

        return pooled.view(-1)

    def forward(self, target_global, target_inter_states, adv_global, adv_inter_states):
        total_loss = 0.0
        if self.use_global:
            loss_global = -F.cosine_similarity(target_global, adv_global, dim=-1).mean()
            total_loss += loss_global
        if self.use_intermediate:
            loss_fine_grained = 0.0
            
            for layer_idx in self.target_layers:
                target_patches = target_inter_states[layer_idx][:, 1:, :] 
                adv_patches = adv_inter_states[layer_idx][:, 1:, :]

                target_patches = F.normalize(target_patches, p=2, dim=-1)
                adv_patches = F.normalize(adv_patches, p=2, dim=-1)

                sim = F.cosine_similarity(adv_patches, target_patches, dim=-1)

                if self.use_spatial_mask:
                    mask_weight = self.spatial_mask.unsqueeze(0)
                    sim = sim * mask_weight
                    layer_sim = sim.sum(dim=-1) / self.spatial_mask.sum()
                else:
                    layer_sim = sim.mean(dim=-1)
                loss_fine_grained -= layer_sim.mean() 
            
            total_loss += self.lambda_fg * loss_fine_grained

        return total_loss