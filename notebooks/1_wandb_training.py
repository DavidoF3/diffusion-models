from types import SimpleNamespace
from pathlib import Path
from tqdm.notebook import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from utils.wandb_utils import *

import wandb

# The content of this script is the same as that in: wandb_training.ipynb  !!!!!!

# Setup --------------------------------------------------

# we are storing the parameters to be logged to wandb
DATA_DIR = Path('../data_train/')
SAVE_DIR = Path('../weights/')
SAVE_DIR.mkdir(exist_ok=True, parents=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = SimpleNamespace(
    # hyperparameters
    num_samples = 30,

    # diffusion hyperparameters
    timesteps = 500,
    beta1 = 1e-4,
    beta2 = 0.02,

    # network hyperparameters
    n_feat = 64, # 64 hidden dimension feature
    n_cfeat = 5, # context vector is of size 5
    height = 16, # 16x16 image
    
    # training hyperparameters
    batch_size = 100,
    n_epoch = 32,
    lrate = 1e-3,
)


# DDPM setuo --------------------------------------------

# Setup ddpm sampler functions
# - perturb_input: Adds noise to the input image at the corresponding timestep on the schedule
# - sample_ddpm_context: Generate images using the DDPM sampler, we will use this function during 
#                        training to sample from the model regularly and see how our training is progressing
perturb_input, sample_ddpm_context = setup_ddpm(config.beta1, 
                                                config.beta2, 
                                                config.timesteps, 
                                                DEVICE)


# Create NN ---------------------------------------------

# construct model
nn_model = ContextUnet(in_channels=3, 
                       n_feat=config.n_feat, 
                       n_cfeat=config.n_cfeat, 
                       height=config.height).to(DEVICE)


# Data loading -----------------------------------------

# load dataset and construct optimizer
dataset = CustomDataset.from_np(path=DATA_DIR)
dataloader = DataLoader(dataset, 
                        batch_size=config.batch_size, 
                        shuffle=True)
optim = torch.optim.Adam(nn_model.parameters(), lr=config.lrate)


# Training --------------------------------------------

# Noise vector
# x_T ~ N(0, 1), sample initial noise
# - we choose noise only once for consistency
noises = torch.randn(config.num_samples, 3, 
                     config.height, config.height).to(DEVICE)  

# A fixed context vector to sample from
ctx_vector = F.one_hot(torch.tensor([0,0,0,0,0,0,   # hero
                                     1,1,1,1,1,1,   # non-hero
                                     2,2,2,2,2,2,   # food
                                     3,3,3,3,3,3,   # spell
                                     4,4,4,4,4,4]), # side-facing 
                       5).to(DEVICE).float()


# Create a wandb run - to track training
# - job_type: helps to identify this job later on
run = wandb.init(project="diff_model_sprite", 
                 job_type="train",   
                 config=config)

# Pass the config back from W&B
config = wandb.config

for ep in tqdm(range(config.n_epoch), leave=True, total=config.n_epoch):
    # Set model to train mode
    nn_model.train()
    optim.param_groups[0]['lr'] = config.lrate*(1-ep/config.n_epoch)
    
    pbar = tqdm(dataloader, leave=False)
    for x, c in pbar:   # x: images  c: context
        # Set gradients to zero for every mini-batch training iteration
        optim.zero_grad()
        x = x.to(DEVICE)
        c = c.to(DEVICE)   

        # Randomly mask out c
        context_mask = torch.bernoulli(torch.zeros(c.shape[0]) + 0.8).to(DEVICE)
        # - with some randomness, we completely mask out the context so that 
        #   the model is able to generally learn what a sprite is as well
        c = c * context_mask.unsqueeze(-1)   

        # Perturb data
        # - perturb_input: Adds noise to the input image at the corresponding timestep on the schedule
        noise = torch.randn_like(x)
        t = torch.randint(1, config.timesteps + 1, (x.shape[0],)).to(DEVICE) 
        x_pert = perturb_input(x, t, noise)  

        # Use network to recover noise    
        pred_noise = nn_model(x_pert, t / config.timesteps, c=c)   

        # Loss: mean squared error between the predicted and true noise   
        loss = F.mse_loss(pred_noise, noise)

        # Back propagation to compute gradients
        loss.backward()    
        # Update weights
        optim.step()

        # Log parameters at the end of each mini-batch iteration
        wandb.log({"loss": loss.item(),
                   "lr": optim.param_groups[0]['lr'],
                   "epoch": ep})


    # save model every 4 epochs
    if ep%4==0 or ep == int(config.n_epoch-1):
        nn_model.eval()
        ckpt_file = SAVE_DIR/f"context_model.pth"
        torch.save(nn_model.state_dict(), ckpt_file)

        # Create W&B artifact - to record model weights
        artifact_name = f"{wandb.run.id}_context_model"
        at = wandb.Artifact(artifact_name, type="model")
        at.add_file(ckpt_file)
        wandb.log_artifact(at, aliases=[f"epoch_{ep}"])

        # sample_ddpm_context: Generate images using the DDPM sampler
        samples, _ = sample_ddpm_context(nn_model, 
                                         noises, 
                                         ctx_vector[:config.num_samples])
        
        # Sample images to view in W&B workspace
        wandb.log({
            "train_samples": [
                wandb.Image(img) for img in samples.split(1)
            ]})
        
# finish W&B run
wandb.finish()