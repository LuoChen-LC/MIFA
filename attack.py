import torch
import torch.nn.functional as F
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
import json
import hashlib

from utils.dataset import ImagePairDataset
from torch.utils.data import DataLoader
from model.clip.ClipB32 import ClipB32FeatureExtractor, FineGrainedCLIPLoss
#from model.clip.ClipB16 import ClipB16FeatureExtractor, FineGrainedCLIPLoss
import os
from datetime import datetime
import torchvision

HASH_FIELDS = [
    "data",
    "model",
    "optim",
    "attack",
]
def config_hash(cfg):
    selected = {}

    for field in HASH_FIELDS:
        value = OmegaConf.select(cfg, field)
        
        selected[field] = OmegaConf.to_container(value, resolve=True)

    cfg_str = json.dumps(selected, sort_keys=True)
    h = hashlib.md5(cfg_str.encode()).hexdigest()
    print(h)
    return h

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    # step 1: get config
    # step 2: dataloader
    dataset = ImagePairDataset(config)
    dataloader = DataLoader(dataset, batch_size=config.data.batch_size, shuffle=False)
    # step 3: model
    model = ClipB32FeatureExtractor()
    model.to(config.model.device)
    # step 4: attack
    attack_imgpair(config, model, dataloader)


def attack_imgpair(config: DictConfig, model, dataloader):

    attack_fn = {
        "fgsm": fgsm_attack, 
        "pgd": pgd_attack,
        "mi-fgsm": mi_fgsm_attack
    }[config.attack]
    save_dir = config.data.output
    save_dir = os.path.join(config.data.output, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), config.data.dataset)


    os.makedirs(save_dir, exist_ok=True)

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Batch Processing",total=config.data.num_samples)):
        if config.data.batch_size * (batch_idx + 1) > config.data.num_samples:
            break
        print(f"\nProcessing image {batch_idx * config.data.batch_size + 1}/{config.data.num_samples}")

        source_img = batch["attacker"]
        target_img = batch["victim"]
        attacker_name = batch["attacker_name"][0]  

        adv_img = attack_fn(config, model, source_img, target_img)
        
        file_name = os.path.basename(attacker_name)
        save_path = os.path.join(save_dir, file_name)
        torchvision.utils.save_image(adv_img[0], save_path)


def fgsm_attack(config, model, source_img, target_img):

    model.eval()

    loss_fn = FineGrainedCLIPLoss(use_global=config.model.use_global, use_intermediate=config.model.use_intermediate, use_spatial_mask=config.model.use_spatial_mask).to(config.model.device)

    device = config.model.device
    source_img = source_img.to(device)
    target_img = target_img.to(device)
    
    delta = torch.zeros_like(source_img, requires_grad=True)
    
    with torch.no_grad():
        clean_global, clean_hidden = model(target_img)

    pbar = tqdm(range(config.optim.steps), desc="Attacking")

    for epoch in pbar:
        adv_img = source_img + delta

        adv_global, adv_hidden = model(adv_img)

        loss = loss_fn(clean_global, clean_hidden, adv_global, adv_hidden)

        grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]

        with torch.no_grad():
            delta.data = delta.data - config.optim.alpha * torch.sign(grad)
            delta.data = torch.clamp(
                delta.data, min=-config.optim.epsilon, max=config.optim.epsilon
            )

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    adv_img = source_img + delta
    adv_img = torch.clamp(adv_img / 255.0, min=0.0, max=1.0).detach()

    return adv_img

def pgd_attack(config, model, source_img, target_img):

    model.eval()

    loss_fn = FineGrainedCLIPLoss(use_global=config.model.use_global, use_intermediate=config.model.use_intermediate, use_spatial_mask=config.model.use_spatial_mask).to(config.model.device)

    device = config.model.device
    source_img = source_img.to(device)
    target_img = target_img.to(device)
    
    delta = torch.zeros_like(source_img, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=config.optim.alpha)

    with torch.no_grad():
        clean_global, clean_hidden = model(target_img)

    pbar = tqdm(range(config.optim.steps), desc="Attacking")

    for epoch in pbar:
        adv_img = source_img + delta

        adv_global, adv_hidden = model(adv_img)

        loss = loss_fn(clean_global, clean_hidden, adv_global, adv_hidden)

        optimizer.zero_grad()
        loss.backward()

        # PGD update
        optimizer.step()
        delta.data = torch.clamp(
            delta,
            min=-config.optim.epsilon,
            max=config.optim.epsilon,
        )

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    adv_img = source_img + delta
    adv_img = torch.clamp(adv_img / 255.0, min=0.0, max=1.0).detach()

    return adv_img

def mi_fgsm_attack(config, model, source_img, target_img):
    model.eval()
    loss_fn = FineGrainedCLIPLoss(use_global=config.model.use_global, use_intermediate=config.model.use_intermediate, use_spatial_mask=config.model.use_spatial_mask).to(config.model.device)
    device = config.model.device
    source_img = source_img.to(device)
    target_img = target_img.to(device)
    
    delta = torch.zeros_like(source_img, requires_grad=True)
    momentum = torch.zeros_like(source_img).detach()
    
    decay_factor = getattr(config.optim, 'mu', 1.0) 
    
    with torch.no_grad():
        target_global, target_hidden = model(target_img)

    pbar = tqdm(range(config.optim.steps), desc="MI-FGSM Attacking")

    for epoch in pbar:
        adv_img = source_img + delta

        adv_global, adv_hidden = model(adv_img)
        loss = loss_fn(target_global, target_hidden, adv_global, adv_hidden)

        grad = torch.autograd.grad(loss, delta, retain_graph=False)[0]

        with torch.no_grad():
            grad_l1_norm = torch.norm(grad, p=1, dim=[1, 2, 3], keepdim=True)
            grad_normalized = grad / (grad_l1_norm + 1e-8)
            
            momentum = decay_factor * momentum + grad_normalized

            delta.data = delta.data - config.optim.alpha * torch.sign(momentum)
            
            delta.data = torch.clamp(delta.data, min=-config.optim.epsilon, max=config.optim.epsilon)
            
            adv_img_tmp = torch.clamp(source_img + delta.data, min=0.0, max=255.0)
            delta.data = adv_img_tmp - source_img

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    adv_img = source_img + delta
    adv_img = torch.clamp(adv_img / 255.0, min=0.0, max=1.0).detach()

    return adv_img
if __name__ == "__main__":
    main()


