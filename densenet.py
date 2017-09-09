from base_model import BaseModel
from layers import ConvolutionalLayer, BatchNormLayer, FullyConnectedLayer, DenseBlock, PoolingLayer, BatchNormDNNLayer


class DenseNet(BaseModel):
    def __init__(self, config_file, **kwargs):
        super(DenseNet, self).__init__(config_file, **kwargs)

        self.input_shape = (None, self.config['data']['input_shape'][2], self.config['data']['input_shape'][0],
                            self.config['data']['input_shape'][1])
        self.output_shape = self.config['data']['output_shape']
        self.augmentation = self.config['data']['augmentation']

        self.growth_rate = kwargs.get('growth_rate', 12)
        self.first_output = kwargs.get('first_output', 16)
        self.num_blocks = kwargs.get('num_blocks', 3)
        self.depth = kwargs.get('depth', 40)
        self.dropout = kwargs.get('dropout', 0)

        self.model.append(ConvolutionalLayer(self.input_shape, (self.first_output, self.input_shape[1], 3, 3),
                                             He_init='normal', He_init_gain='relu', activation='linear',
                                             layer_name='pre_conv'))
        n = (self.depth - 1) // self.num_blocks
        for b in xrange(self.num_blocks):
            self.model.append(DenseBlock(self.model[-1].get_output_shape(), num_conv_layer=n - 1,
                                         growth_rate=self.growth_rate, dropout=self.dropout,
                                         layer_name='dense_block_%d' % b))
            if b < self.num_blocks - 1:
                self.model.append(DenseBlock(self.model[-1].get_output_shape(), True, None, None, self.dropout,
                                             'dense_block_%d' % b))

        self.model.append(BatchNormLayer(self.model[-1].get_output_shape(), layer_name='post_bn'))
        shape = self.model[-1].get_output_shape()
        self.model.append(PoolingLayer(shape, (shape[2], shape[3]), stride=(1, 1), mode='average_exc_pad',
                                       layer_name='post_pool'))
        self.model.append(FullyConnectedLayer(self.model[-1].get_output_shape(True)[1], self.output_shape,
                                              He_init='normal', He_init_gain='softmax', layer_name='softmax',
                                              activation='softmax', target='dev1'))
        self.params = [p for layer in self.model for p in layer.params]
        self.regularizable = [p for layer in self.model for p in layer.regularizable]

    def inference(self, input, training=False):
        super(DenseNet, self).set_training_status(training)
        return super(DenseNet, self).inference(input)
