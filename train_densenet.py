import numpy as np
import theano
from theano import tensor as T

from neuralnet import metrics
from neuralnet import monitor
from densenet import DenseNet
from densenet import DataManager2


def train(config_file, **kwargs):
    x = T.tensor4('input', theano.config.floatX)
    y = T.ivector('output')

    net = DenseNet(config_file)
    mon = monitor.Monitor(config_file)
    mon.dump_model(net)

    shape = (net.batch_size, net.input_shape[1], net.input_shape[2], net.input_shape[3])
    placeholder_x = theano.shared(np.zeros(shape, 'float32'), 'input_placeholder')
    placeholder_y = theano.shared(np.zeros((net.batch_size,), 'int32'), 'label_placeholder')
    placeholder_lr = theano.shared(np.cast[theano.config.floatX](net.learning_rate), 'learning_rate')

    net.set_training_status(True)
    p_y_given_x_train = net.inference(x)
    cost = net.build_cost(p_y_given_x_train, y, **{'params': net.regularizable})
    updates = net.build_updates(cost, net.trainable, **{'learning_rate': placeholder_lr})
    train_network = net.compile([], cost, updates=updates, givens={x: placeholder_x, y: placeholder_y}, name='train_densenet', allow_input_downcast=True)

    net.set_training_status(False)
    p_y_given_x_test = net.inference(x)
    cost = net.build_cost(p_y_given_x_test, y, **{'params': net.regularizable})
    accuracy = (1. - metrics.mean_classification_error(p_y_given_x_test, y)) * 100.
    test_network = net.compile([], [cost, accuracy], givens={x: placeholder_x, y: placeholder_y}, name='test_densenet', allow_input_downcast=True)

    dm = DataManager2(config_file, (placeholder_x, placeholder_y))
    epoch = 0
    num_training_batches = dm.num_train_data // net.batch_size
    print('Training...')
    while epoch < net.n_epochs:
        epoch += 1
        if epoch == (net.n_epochs // 2) or epoch == (net.n_epochs * 3 // 4):
            placeholder_lr.set_value(placeholder_lr.get_value() * np.float32(0.1))
            print('Learning rate decreased to %.10f' % placeholder_lr.get_value())

        batches = dm.get_batches(epoch, net.n_epochs, True, False, 0.5, 4)
        idx = 0
        for b in batches:
            iteration = (epoch - 1.) * num_training_batches + idx + 1

            x, y = b
            dm.update_input((x, y))
            training_cost = train_network()
            if np.isnan(training_cost):
                raise ValueError('Training failed due to NaN cost')
            mon.plot('training cost', training_cost)

            if iteration % net.validation_frequency == 0:
                batch_valid = dm.get_batches(training=False)

                for b_valid in batch_valid:
                    dm.update_input((b_valid[0], b_valid[1]))
                    c, a = test_network()
                    mon.plot('validation cost', c)
                    mon.plot('validation accuracy', a)
                mon.flush()
                net.save_params()
            idx += 1
            mon.tick()
    mon.flush()

    print('Training the network with all available data...')
    data = (np.concatenate((dm.training_set[0], dm.testing_set[0])),
            np.concatenate((dm.training_set[1], dm.testing_set[1])))
    dm.training_set = data
    dm.num_train_data = data[0].shape[0] / net.batch_size

    net.reset()
    mon.reset()
    placeholder_lr.set_value(np.cast[theano.config.floatX](net.learning_rate))

    for i in range(net.n_epochs):
        print('Epoch %d starts...' % (i + 1))
        if i + 1 == (net.n_epochs // 2) or i + 1 == (net.n_epochs * 3 // 4):
            placeholder_lr.set_value(placeholder_lr.get_value() * np.float32(0.1))
            print('Learning rate decreased to %.10f' % placeholder_lr.get_value())
        batches = dm.get_batches(i + 1, net.n_epochs, False, 0.5, 4)
        for b in batches:
            x, y = b
            dm.update_input((x, y))
            c = train_network()
            mon.plot('retraining cost', c)
            mon.tick()
    mon.flush()
    net.save_params()


if __name__ == '__main__':
    train('densenet.config')
