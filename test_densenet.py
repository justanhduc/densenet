import theano
from theano import tensor as T
from densenet import DenseNet
import numpy as np

import metrics
import utils


def test_batches_with_labels(config_file, **kwargs):
    test_data = kwargs.get('testing_data')

    x = T.tensor4('input', theano.config.floatX)
    y = T.ivector('output')

    net = DenseNet(config_file)
    net.load_params()
    shape = (net.testing_batch_size, net.input_shape[1], net.input_shape[2], net.input_shape[3])
    placeholder_x = theano.shared(np.zeros(shape, 'float32'), 'input_placeholder')
    placeholder_y = theano.shared(np.zeros((net.testing_batch_size,), 'int32'), 'label_placeholder')

    p_y_given_x_test = net.inference(x)
    cost = net.build_cost(p_y_given_x_test, y, **{'params': net.regularizable})
    accuracy = (1. - metrics.MeanClassificationErrors(p_y_given_x_test, y)) * 100.
    test_network = net.compile([], [cost, accuracy], givens={x: placeholder_x, y: placeholder_y}, name='test_densenet', allow_input_downcast=True)

    num_test_batches = test_data[0].shape[0] / net.testing_batch_size
    cost = 0.
    accuracy = 0.
    data_manager_test = utils.DataManager(test_data, net.testing_batch_size, (placeholder_x, placeholder_y))
    batches = data_manager_test.get_batches()
    print 'Testing %d batches...' % num_test_batches
    for x, y in batches:
        data_manager_test.update_input((x, y))
        c, a = test_network()
        cost += c
        accuracy += a
    cost /= num_test_batches
    accuracy /= num_test_batches
    print 'Testing finished. Testing cost is %.2f. Testing accuracy is %.2f %%.' % (cost, accuracy)


def test_image(config_file, image):
    image = np.array(image)
    assert len(image.shape) == 3 and image.shape[-1] == 3, 'Testing image must be an RGB color image.'

    x = T.tensor4('input', theano.config.floatX)
    net = DenseNet(config_file)
    net.load_params()
    p_y_given_x_test = net.inference(x)
    prediction = T.argmax(p_y_given_x_test, 1) if p_y_given_x_test.ndim > 1 else p_y_given_x_test >= 0.5
    test_network = net.compile([x], prediction, name='test_densenet', allow_input_downcast=True)
    prediction = test_network(np.expand_dims(image, 0).transpose((0, 3, 1, 2)))
    print 'The image belongs to class %d' % prediction


if __name__ == '__main__':
    import read_data
    _, _, X_test, y_test = read_data.load_dataset('C:\Users\just.anhduc\Downloads')
    kwargs = {'testing_data': (X_test, y_test)}
    test_batches_with_labels('densenet.config', **kwargs)
