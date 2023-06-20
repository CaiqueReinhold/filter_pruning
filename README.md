# Evolutionary Pruning of Deep Convolutional Neural Networks

Implementation of the evolutionary pruning algorithm described in the paper [
Filter Pruning for Efficient Transfer Learning in Deep Convolutional Neural Networks](https://link.springer.com/chapter/10.1007/978-3-030-20912-4_19).

The method uses open source pre trained image recognition models and prunes convolutional filters using an evolutionary algorithm. The pruned models are then fine tuned on the target dataset.

The objetive is to use pre trained models to perform transfer learning on a target dataset, while reducing the computational cost of the model.

Results on Flowers-102 dataset:
 - AlexNet: 43.02% FLOPS reduction, no accuracy loss
 - VGG-16: 64.4% FLOPS reduction, no accuracy loss 

Results on Caltech-256 dataset:
 - AlexNet: 34.14% FLOPS reduction, 1.56% accuracy loss
 - VGG-16: 39.9% FLOPS reduction, no accuracy loss