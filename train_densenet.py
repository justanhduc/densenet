from densenet import DenseNet
import metrics
import utils

from matplotlib import pyplot as plt
import theano
from theano import tensor as T
import numpy as np
import time


def train(config_file, **kwargs):
    training_data = kwargs.get('training_data')
    validation_data = kwargs.get('validation_data')

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

    # p_y_given_x_test = net.inference(x)
    # cost = net.build_cost(p_y_given_x_test, y, **{'params': net.regularizable})
    # accuracy = (1. - metrics.MeanClassificationErrors(p_y_given_x_test, y)) * 100.
    # test_network = net.compile([], [cost, accuracy], givens={x: placeholder_x, y: placeholder_y}, name='test_densenet', allow_input_downcast=True)
    #
    # epoch = 0
    # vote_to_terminate = 0
    # num_training_batches = training_data[0].shape[0] / net.batch_size
    # num_validation_batches = validation_data[0].shape[0] / net.validation_batch_size
    # best_accuracy = 0.
    # best_epoch = 0
    # if net.display_cost:
    #     training_cost_to_plot = []
    #     validation_cost_to_plot = []
    # print 'Training...'
    #
    # data_manager_train = utils.DataManager(training_data, net.batch_size, (placeholder_x, placeholder_y), True, False,
    #                                        net.augmentation)
    # data_manager_valid = utils.DataManager(validation_data, net.validation_batch_size, (placeholder_x, placeholder_y))
    start_training_time = time.time()
    # while epoch < net.n_epochs:
    #     epoch += 1
    #     if epoch == (net.n_epochs // 2) or epoch == (net.n_epochs * 3 // 4):
    #         placeholder_lr.set_value(placeholder_lr.get_value() * np.float32(0.1))
    #         print '\tlearning rate decreased to %.10f' % placeholder_lr.get_value()
    #     training_cost = 0.
    #     start_epoch_time = time.time()
    #     batches = data_manager_train.get_batches(epoch, net.n_epochs)
    #     idx = 0
    #     for b in batches:
    #         iteration = (epoch - 1.) * num_training_batches + idx + 1
    #
    #         x, y = b
    #         data_manager_train.update_input((x, y))
    #         training_cost += train_network()
    #         if np.isnan(training_cost):
    #             raise ValueError('Training failed due to NaN cost')
    #
    #         if iteration % net.validation_frequency == 0:
    #             batch_valid = data_manager_valid.get_batches()
    #             validation_cost = 0.
    #             validation_accuracy = 0.
    #             for b_valid in batch_valid:
    #                 data_manager_valid.update_input((b_valid[0], b_valid[1]))
    #                 c, a = test_network()
    #                 validation_cost += c
    #                 validation_accuracy += a
    #             validation_cost /= num_validation_batches
    #             validation_accuracy /= num_validation_batches
    #             print '\tvalidation cost: %.4f' % validation_cost
    #             print '\tvalidation accuracy: %.4f' % validation_accuracy
    #             if validation_accuracy > best_accuracy:
    #                 best_epoch = epoch
    #                 best_accuracy = validation_accuracy
    #                 vote_to_terminate = 0
    #                 print '\tbest validation accuracy: %.4f' % best_accuracy
    #                 if net.extract_params:
    #                     net.save_params()
    #             else:
    #                 vote_to_terminate += 1
    #
    #             if net.display_cost:
    #                 training_cost_to_plot.append(training_cost / (idx + 1))
    #                 validation_cost_to_plot.append(validation_cost)
    #                 plt.clf()
    #                 plt.plot(training_cost_to_plot)
    #                 plt.plot(validation_cost_to_plot)
    #                 plt.show(block=False)
    #                 plt.pause(1e-5)
    #         idx += 1
    #     training_cost /= num_training_batches
    #     print '\tepoch %d took %.2f mins' % (epoch, (time.time() - start_epoch_time) / 60.)
    #     print '\ttraining cost: %.4f' % training_cost
    # if net.display_cost:
    #     plt.savefig('%s/training_curve.png' % net.save_path)
    # print 'Best validation accuracy: %.4f' % best_accuracy

    print 'Training the network with all available data...'
    data = (np.concatenate((training_data[0], validation_data[0])), np.concatenate((training_data[1], validation_data[1])))
    num_training_batches = data[0].shape[0] / net.batch_size

    net.reset()
    placeholder_lr.set_value(np.cast[theano.config.floatX](net.learning_rate))

    data_manager_train = utils.DataManager(data, net.batch_size, (placeholder_x, placeholder_y), True, False,
                                           net.augmentation)
    for i in range(net.n_epochs):
        print 'Epoch %d starts...' % (i + 1)
        if i + 1 == (net.n_epochs // 2) or i + 1 == (net.n_epochs * 3 // 4):
            placeholder_lr.set_value(placeholder_lr.get_value() * np.float32(0.1))
            print '\tlearning rate decreased to %.10f' % placeholder_lr.get_value()
        batches = data_manager_train.get_batches(i + 1, net.n_epochs)
        training_cost = 0.
        for b in batches:
            x, y = b
            data_manager_train.update_input((x, y))
            training_cost += train_network()
        training_cost /= num_training_batches
        print '\ttraining cost: %.4f' % training_cost
    net.save_params()
    print 'Training ended after %.2f hours' % ((time.time() - start_training_time) / 3600.)


if __name__ == '__main__':
    import read_data
    X_train, y_train, _, _ = read_data.load_dataset('C:\Users\just.anhduc\Downloads')
    X_val, y_val = X_train[-5000:], y_train[-5000:]
    X_train, y_train = X_train[:-5000], y_train[:-5000]
    kwargs = {'training_data': (X_train, y_train), 'validation_data': (X_val, y_val)}
    train('densenet.config', **kwargs)
