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
    shape = (net.batch_size, net.input_shape[1], net.input_shape[2], net.input_shape[3])
    placeholder_x = theano.shared(np.zeros(shape, 'float32'), 'input_placeholder')
    placeholder_y = theano.shared(np.zeros((net.batch_size,), 'int32'), 'label_placeholder')
    placeholder_lr = theano.shared(np.cast[theano.config.floatX](net.learning_rate), 'learning_rate')

    p_y_given_x_train = net.inference(x, True)
    cost = net.build_cost(p_y_given_x_train, y, **{'params': net.regularizable})
    updates = net.build_updates(cost, net.trainable, **{'learning_rate': placeholder_lr})
    train_network = net.compile([], cost, updates=updates, givens={x: placeholder_x, y: placeholder_y}, name='train_densenet', allow_input_downcast=True)

    p_y_given_x_test = net.inference(x)
    cost = net.build_cost(p_y_given_x_test, y, **{'params': net.regularizable})
    accuracy = (1. - metrics.MeanClassificationErrors(p_y_given_x_test, y)) * 100.
    test_network = net.compile([], [cost, accuracy], givens={x: placeholder_x, y: placeholder_y}, name='test_densenet', allow_input_downcast=True)

    data_manager = DataManager2(config_file, (placeholder_x, placeholder_y))
    mon = monitor.Monitor(config_file)
    epoch = 0
    vote_to_terminate = 0
    num_training_batches = data_manager.num_train_data // net.batch_size
    num_validation_batches = data_manager.num_test_data // net.validation_batch_size
    best_accuracy = 0.
    best_epoch = 0
    print('Training...')
    while epoch < net.n_epochs:
        epoch += 1
        if epoch == (net.n_epochs // 2) or epoch == (net.n_epochs * 3 // 4):
            placeholder_lr.set_value(placeholder_lr.get_value() * np.float32(0.1))
            print('Learning rate decreased to %.10f' % placeholder_lr.get_value())

        batches = data_manager.get_batches(epoch, net.n_epochs, 'train', 0.5, 4)
        idx = 0
        for b in batches:
            iteration = (epoch - 1.) * num_training_batches + idx + 1

            x, y = b
            data_manager.update_input((x, y))
            training_cost = train_network()
            if np.isnan(training_cost):
                raise ValueError('Training failed due to NaN cost')
            mon.plot('training cost', training_cost)

            if iteration % net.validation_frequency == 0:
                batch_valid = data_manager.get_batches(stage='test')
                validation_cost = 0.
                validation_accuracy = 0.
                for b_valid in batch_valid:
                    data_manager.update_input((b_valid[0], b_valid[1]))
                    c, a = test_network()
                    validation_cost += c
                    validation_accuracy += a
                    mon.plot('validation cost', c)
                    mon.plot('validation accuracy', a)
                validation_cost /= num_validation_batches
                validation_accuracy /= num_validation_batches

                if validation_accuracy > best_accuracy:
                    best_epoch = epoch
                    best_accuracy = validation_accuracy
                    vote_to_terminate = 0
                    if net.extract_params:
                        net.save_params()
                else:
                    vote_to_terminate += 1
                mon.flush()
            idx += 1
            mon.tick()
    print('Best validation accuracy: %.4f at epoch %d' % (best_accuracy, best_epoch))

    print('Training the network with all available data...')
    data = (np.concatenate((data_manager.training_set[0], data_manager.testing_set[0])),
            np.concatenate((data_manager.training_set[1], data_manager.testing_set[1])))
    data_manager.training_set = data
    data_manager.num_train_data = data[0].shape[0] / net.batch_size

    net.reset()
    placeholder_lr.set_value(np.cast[theano.config.floatX](net.learning_rate))

    for i in range(best_epoch):
        print('Epoch %d starts...' % (i + 1))
        if i + 1 == (net.n_epochs // 2) or i + 1 == (net.n_epochs * 3 // 4):
            placeholder_lr.set_value(placeholder_lr.get_value() * np.float32(0.1))
            print('Learning rate decreased to %.10f' % placeholder_lr.get_value())
        batches = data_manager.get_batches(i + 1, net.n_epochs, 0.5, 4)
        training_cost = 0.
        for b in batches:
            x, y = b
            data_manager.update_input((x, y))
            training_cost += train_network()
        training_cost /= num_training_batches
        print('\ttraining cost: %.4f' % training_cost)
    net.save_params()


if __name__ == '__main__':
    train('densenet.config')
