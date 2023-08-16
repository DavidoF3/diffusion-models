# diffusion-models

This repository describes the theory behind diffusion models and implements several variations of a diffusion model.

Repository content list:

- [diffusion-models](#diffusion-models)
- [Repository structure](#repository-structure)
- [Theory - diffusion models](#theory---diffusion-models)
  - [Inference: sampling](#inference-sampling)
  - [NN architecture](#nn-architecture)
  - [Training](#training)
  - [Controlling NN](#controlling-nn)
  - [Speeding up diffusion models](#speeding-up-diffusion-models)
- [Run](#run)
- [References](#references)

# Repository structure

The subfolder [notebooks](/notebooks/) contains the code in this repository. 
* The notebook [basics_training_and_inference.ipynb](/notebooks/0_basics_training_and_inference.ipynb) goes through: 1) inference of a pre-trained diffusion model (using DDPM sampling), 2) training of a diffusion model without and with context embedding (using DDPM sampling), 3) inference of a pre-trained diffusion model (using DDIM sampling).
* The notebook [wandb_training.ipynb](/notebooks/1_wandb_training.ipynb) goes through: the steps to track the training progress of a diffusion model by using `W&B`. A context embedding and DDPM sampling are used during training.
* The notebook [wandb_inference.ipynb](/notebooks/2_wandb_inference.ipynb) goes through: 1) pulling a diffusion model registered in the W&B model registry, 2) running two different sampling methods (DDPM and DDIM) on a range of noise inputs, 3) uploading the results to W&B using W&B Tables, 4) comparing the sampling results (see [report](https://wandb.ai/doc93/diff_model_sprite/reports/Sampling-DDPM-vs-DDIM--Vmlldzo1MTQ5OTQ5)).

The subfolder [utils](/utils/)  contains supporting functions and classes used by the notebooks.


# Theory - diffusion models

## Inference: sampling

Diffusion models attempt to predict the noise of an image. This noise is then subtracted from the image to obtain the requested output. A single pass of this procedure is not enough to remove all the noise. Multiple iterations are run to "fully" remove the noise and obtain a good quality inference result.

```Iteratively move from a fully diffused (noisy) image to a clear image```

NN expects as input a noisy sample with normally distributed noise. After subtracting the predicted noise from the sample, the noise in the sample is no longer normally distributed. Hence, before moving to the next inference iteration, we must add some noise to the sample (scaled based on the time step of inference), to make the noise in the sample normally distributed again. 

`Denoising Diffusion Probabilistic Model (DDPM)`: this is a sampling algorithm used for subtracting noise from the image + add noise back (in line with above explanation).

More details on denoising are given in [Section: Speeding up diffusion models](#speeding-up-diffusion-models).

## NN architecture

A UNet architecture can be used to build a diffusion model. The output noise matrix from this architecture has the same number of dimensions as the input sample.

Embeddings can be used to pass information into the NN at the output from each decoder upsampling level:
* Time embedding: related to noise scaling. 
  * Add it into the upsampling (decoder) of the UNet.
* Context embedding: for controlling the generation (eg. with a text description)
  * Multiply it in the upsampling (decoder) of the UNet.

```
up_block_2out = self.up_block_1_func(context_emb * up_block_1out + t_emb, down_block_2out)
```

## Training

During training, the NN learns the distribution of what is not noise i.e. learns to predict noise in the image. To do so:

1. Take a denoised image (gt)
2. Generate some noise (gt) (noise level determined by samplinga time step)
3. Add noise to the gt image
4. Feed image with noise to NN and NN returns the noise map
5. Loss = predicted noise - gt noise
6. Backpropagate through NN and optimise

To improve training stability, add different noise levels (i.e. by sampling a random time step) per image accross an epoch.

## Controlling NN

As briefly discussed in [Section: NNarchitecture](#nn-architecture), embeddings are used to contol the output (generation) from the model. Embeddings are vectors that capture meaning in a latent space. Hence, for exampla, text with similar meaning will have similar vector embeddings (smaller distance between them).

During training we can add context to the noise prediction process by feeding a context embedding in addition to the image with noise. Hence, the training procedure can be updated to:

1. Take a denoised image (gt)
2. Generate some noise (gt) (noise level determined by samplinga time step)
3. Add noise to the gt image
4. Generate context embedding of sentence describing the image
5. Feed image with noise + context embedding to NN and NN returns the noise map
6. Loss = predicted noise - gt noise
7. Backpropagate through NN and optimise

Examples of context embeddings are:
* Text embeddings (eg. > 1000 in length)
* Category embedding (eg. length 5 one hot encoding vectors [0 1 0 0 0])

## Speeding up diffusion models

Sampling (denoising) is slow with `Denoising Diffusion Probabilistic Model (DDPM)` becuase:
* Many time steps involved
* Each timestep depends on previous one (Makrkov chain process)

New `samplers` address this problem. An example is `Denoising Diffusion Implicit Model (DDIM)`. DDIM is faster because:
* It skips time steps -> yielding rough approximation of final output
* Refines final output with the denoising process

# Run

Create and activate environment
```
python -m venv venv_diffmodel
source venv_diffmodel/bin/activate
```

Install packages
```
pip install -e .
```

Open `notebooks/sampling_and_training.ipynb` and run cells. Notice that:
* Pretrained weight files and training data specified in the notebook are not part of the repo.

# References

References: deeplearning.ai