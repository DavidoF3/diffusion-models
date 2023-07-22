# diffusion-models

This repository describes the theory behind diffusion models and implements several variations of a diffusion model.

Repository content list:

- [diffusion-models](#diffusion-models)
- [Theory](#theory)
  - [Inference: sampling](#inference-sampling)
  - [NN architecture](#nn-architecture)
  - [Training](#training)
  - [Controlling NN](#controlling-nn)
  - [Speeding up diffusion models](#speeding-up-diffusion-models)
- [Run](#run)
- [References](#references)


# Theory

## Inference: sampling

Diffusion models attempt to predict the noise of an image. This noise is then subtracted from the image to obtain the requested output. A single pass of this procedure is not enough to remove all the noise. Multiple iterations are run to "fully" remove the noise and obtain a good quality inference result.

```Iteratively move from a fully diffused (noisy) image to a clear image```

NN expects as input a noisy sample with normally distributed noise. After subtracting the predicted noise from the sample, the noise in the sample is no longer normally distributed. Hence, before moving to the next inference iteration, we must add some noise to the sample (scaled based on the time step of inference), to make the noise in the sample normally distributed again.

`Denoising Diffusion Probabilistic Model (DDPM)`: sampling algorithm used for subtracting noise from the image.

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