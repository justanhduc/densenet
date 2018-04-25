from neuralnet.layers import ConvolutionalLayer, BatchNormLayer, FullyConnectedLayer, DenseBlock, PoolingLayer, \
    DecorrBatchNormLayer, ActivationLayer
from neuralnet import Model
from neuralnet.utils import DataManager
from neuralnet import read_data

import numpy as np


class DenseNet(Model):
    def __init__(self, config_file, **kwargs):
        super(DenseNet, self).__init__(config_file, **kwargs)

        self.input_shape = (None, self.config['model']['input_shape'][2], self.config['model']['input_shape'][0],
                            self.config['model']['input_shape'][1])
        self.output_shape = self.config['model']['output_shape']
        self.augmentation = self.config['model']['augmentation']

        self.growth_rate = self.config['model']['growth_rate']
        self.first_output = self.config['model']['first_output']
        self.num_blocks = self.config['model']['num_blocks']
        self.depth = self.config['model']['depth']
        self.dropout = self.config['model']['dropout']

        self.model.append(ConvolutionalLayer(self.input_shape, self.first_output, 3, He_init='normal', He_init_gain='relu',
                                             activation='linear', layer_name='pre_conv'))
        n = (self.depth - 1) // self.num_blocks
        for b in range(self.num_blocks):
            self.model.append(DenseBlock(self.model.output_shape, num_conv_layer=n - 1, growth_rate=self.growth_rate,
                                         dropout=self.dropout, layer_name='dense_block_%d' % b, normlization='bn'))
            if b < self.num_blocks - 1:
                self.model.append(DenseBlock(self.model.output_shape, True, None, None, self.dropout,
                                             layer_name='dense_block_%d' % b, normlization='bn'))

        self.model.append(BatchNormLayer(self.model.output_shape, activation='linear', layer_name='post_bn'))
        shape = self.model.output_shape
        self.model.append(PoolingLayer(shape, (shape[2], shape[3]), stride=(1, 1), mode='average_exc_pad',
                                       layer_name='post_pool'))
        self.model.append(ActivationLayer(self.model.output_shape, 'relu', 'post_bn_relu'))
        self.model.append(FullyConnectedLayer(self.model.output_shape, self.output_shape, He_init='normal',
                                              He_init_gain='softmax', layer_name='softmax', activation='softmax'))

        super(DenseNet, self).get_all_params()
        super(DenseNet, self).get_trainable()
        super(DenseNet, self).get_regularizable()
        super(DenseNet, self).show()

    def inference(self, input):
        return self.model(input)


class DataManager2(DataManager):
    def __init__(self, config_file, placeholders):
        super(DataManager2, self).__init__(config_file, placeholders)
        self.load_data()

    def load_data(self):
        X_train, y_train, _, _ = read_data.load_dataset(self.path)
        X_val, y_val = X_train[-5000:], y_train[-5000:]
        X_train, y_train = X_train[:-5000], y_train[:-5000]
        self.training_set = (X_train, y_train)
        self.testing_set = (X_val, y_val)
        self.num_train_data = X_train.shape[0]
        self.num_test_data = X_val.shape[0]

    def augment_minibatches(self, minibatches, *args):
        """
        Randomly augments images by horizontal flipping with a probability of
        `flip` and random translation of up to `trans` pixels in both directions.
        """
        flip, trans = args
        for batch in minibatches:
            if self.no_target:
                inputs = batch
            else:
                inputs, targets = batch

            batchsize, c, h, w = inputs.shape
            if flip:
                coins = np.random.rand(batchsize) < flip
                inputs = [inp[:, :, ::-1] if coin else inp
                          for inp, coin in zip(inputs, coins)]
                if not trans:
                    inputs = np.asarray(inputs)
            outputs = inputs
            if trans:
                outputs = np.empty((batchsize, c, h, w), inputs[0].dtype)
                shifts = np.random.randint(-trans, trans, (batchsize, 2))
                for outp, inp, (x, y) in zip(outputs, inputs, shifts):
                    if x > 0:
                        outp[:, :x] = 0
                        outp = outp[:, x:]
                        inp = inp[:, :-x]
                    elif x < 0:
                        outp[:, x:] = 0
                        outp = outp[:, :x]
                        inp = inp[:, -x:]
                    if y > 0:
                        outp[:, :, :y] = 0
                        outp = outp[:, :, y:]
                        inp = inp[:, :, :-y]
                    elif y < 0:
                        outp[:, :, y:] = 0
                        outp = outp[:, :, :y]
                        inp = inp[:, :, -y:]
                    outp[:] = inp
            yield outputs, targets if not self.no_target else outputs
