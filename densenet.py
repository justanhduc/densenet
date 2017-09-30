from base_model import BaseModel
from layers import ConvolutionalLayer, BatchNormLayer, FullyConnectedLayer, DenseBlock, PoolingLayer, BatchNormDNNLayer


class DenseNet(BaseModel):
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
        super(DenseNet, self).get_all_params()
        super(DenseNet, self).get_trainable()
        super(DenseNet, self).get_regularizable()

    def inference(self, input, training=False):
        super(DenseNet, self).set_training_status(training)
        return super(DenseNet, self).inference(input)
