# densenet
A Theano implementation of [DenseNet](https://arxiv.org/abs/1608.06993). The model is trained on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

This implementation is written following the [Lasagne implementation](https://github.com/Lasagne/Recipes/tree/master/papers/densenet)

## Requirements
[Theano v0.9](http://deeplearning.net/software/theano/)

[NeuralNet](https://github.com/justanhduc/neuralnet)

## Result
The model is trained using the scheme described in the paper. The training run for 300 epochs. The Multinoulli cross-entropy loss function is optimized using SGD with Nesterov Momentum. Learning rate is 1e-1 and decreased by 10 after 50% and 75% of the training. Using that scheme, the following training curve can be obtained. Testing accuracy on CIFAR-10 testing set is > 94 %.

![training curve](https://github.com/justanhduc/densenet/blob/master/training_curve.png)

## Usages
To train DenseNet using the current training scheme

```
python train_densenet.py
```

To test DenseNet 
```
python test_densenet.py
```


## Credits
[Lasagne](http://lasagne.readthedocs.io/en/latest/)
