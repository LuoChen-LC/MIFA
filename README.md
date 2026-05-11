# Weaponizing Vision-Language Models: Zero-Shot Transfer-Based Attacks on Black-Box Face Recognition
## Overview
This repository implements generative adversarial perturbations for face recognition. Representative methods for both CLIP-ViT-B/16 and CLIP-ViT-B/32 are provided.
## Requirements
~~~bash
pip install -r requirements.txt
~~~
## Quick Start
### Data Preparation
Place images into `Dataset/CelebA-HQ/` and prepare images pair list as `.csv` file. Modify the `config/config.yaml` and `util/dataset.py`. 
### Atack
~~~python
python attack.py
~~~
