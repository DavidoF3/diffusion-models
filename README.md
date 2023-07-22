# diffusion-models
This repository implements a diffusion model

- [diffusion-models](#diffusion-models)
- [Inference: sampling](#inference-sampling)
- [NN architecture](#nn-architecture)
- [Training](#training)

# Inference: sampling

Diffusion models attempt to predict the noise of an image. This noise is then subtracted from the image to obtain the requested output. A single pass of this procedure is not enough to remove all the noise. Multiple iterations are run to "fully" remove the noise and obtain a good quality inference result.

```Iteratively move from a fully diffused (noisy) image to a clear image```

NN expects as input a noisy sample with normally distributed noise. After subtracting the predicted noise from the sample, the noise in the sample is no longer normally distributed. Hence, before moving to the next inference iteration, we must add some noise to the sample (scaled based on the time step of inference), to make the noise in the sample normally distributed again.

`Denoising Diffusion Probabilistic Model (DDPM)`: sampling algorithm used for subtracting noise from the image.

# NN architecture

A UNet architecture can be used to build a diffusion model. The output noise matrix from this architecture has the same number of dimensions as the input sample.

Embeddings can be used to pass information into the NN at the output from each decoder upsampling level:
* Time embedding: related to noise scaling. 
  * Add it into the upsampling (decoder) of the UNet.
* Context embedding: for controlling the generation (eg. with a text description)
  * Multiply it in the upsampling (decoder) of the UNet.

```
up_block_2out = self.up_block_1_func(context_emb * up_block_1out + t_emb, down_block_2out)
```

# Training

During training, the NN learns the distribution of what is not noise i.e. learns to predict noise in the image. To do so:

* Take a denoised image (gt)
* Generate some noise (gt)
* Add noise to the gt image
* Feed image with noise to NN and NN whould return the noise map
* Loss = predicted noise - gt noise
* Backpropagate through NN

To improve training stability, add different noise levels (i.e. by sampling a random time step) per image accross an epoch.

