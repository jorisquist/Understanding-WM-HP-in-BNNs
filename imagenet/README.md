# Imagenet

This is the Imagenet implementation used in our paper. It is based on the implementation of 
[AdamBNN](https://github.com/liuzechun/AdamBNN), but now with our new filtering based optimizer, torch multiprocessing
and [NVIDIA Dali](https://developer.nvidia.com/dali)

## Step 1

It is only necessary to use step 1 when not using our filtering based optimizer, but instead want to compare with other
two-step training methods.

## Step 2

This contains the training script to train a ReActNet with our gradient filtering method.