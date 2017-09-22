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
    # step = T.scalar('step', 'int32')

    net = DenseNet(config_file, **{'dropout': 0})
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

    early_stopping = False
    epoch = 0
    vote_to_terminate = 0
    num_training_batches = training_data[0].shape[0] / net.batch_size
    num_validation_batches = validation_data[0].shape[0] / net.validation_batch_size
    best_accuracy = 0.
    best_epoch = 0
    training_cost_to_plot = []
    validation_cost_to_plot = []
    print 'Training...'
    start_training_time = time.time()
    while epoch < net.n_epochs and not early_stopping:
        epoch += 1
        if epoch == (net.n_epochs // 2) or epoch == (net.n_epochs * 3 // 4):
            placeholder_lr.set_value(placeholder_lr.get_value() * np.float32(0.1))
            print '\tlearning rate decreased to %.10f' % placeholder_lr.get_value()
        training_cost = 0.
        start_epoch_time = time.time()
        batches = utils.generator(training_data, net.batch_size)
        if net.augmentation:
            batches = utils.augment_minibatches(batches)
            batches = utils.generate_in_background(batches)
        batches = utils.progress(batches, desc='Epoch %d/%d, Batch ' % (epoch, net.n_epochs), total=num_training_batches)
        idx = 0
        for b in batches:
            iteration = (epoch - 1.) * num_training_batches + idx + 1

            x, y = b
            utils.update_input((x, y), (placeholder_x, placeholder_y))
            training_cost += train_network()
            if np.isnan(training_cost):
                raise ValueError('Training failed due to NaN cost')

            if iteration % net.validation_frequency == 0:
                batch_valid = utils.generator(validation_data, net.validation_batch_size)
                validation_cost = 0.
                validation_accuracy = 0.
                for b_valid in batch_valid:
                    utils.update_input((b_valid[0], b_valid[1]), (placeholder_x, placeholder_y))
                    c, a = test_network()
                    validation_cost += c
                    validation_accuracy += a
                validation_cost /= num_validation_batches
                validation_accuracy /= num_validation_batches
                print '\tvalidation cost: %.4f' % validation_cost
                print '\tvalidation accuracy: %.4f' % validation_accuracy
                if validation_accuracy > best_accuracy:
                    best_epoch = epoch
                    best_accuracy = validation_accuracy
                    vote_to_terminate = 0
                    # pkl.dump(model, open(save_file, 'wb'))
                    print '\tbest validation accuracy: %.4f' % best_accuracy
                    if net.extract_params:
                        net.save_params()
                    # print '\tbest model dumped to %s' % save_file
                else:
                    vote_to_terminate += 1

                training_cost_to_plot.append(training_cost / (idx + 1))
                validation_cost_to_plot.append(validation_cost)
                plt.clf()
                plt.plot(training_cost_to_plot)
                plt.plot(validation_cost_to_plot)
                plt.show(block=False)
                plt.pause(1e-5)
            idx += 1
        training_cost /= num_training_batches
        print '\tepoch %d took %.2f mins' % (epoch, (time.time() - start_epoch_time) / 60.)
        print '\ttraining cost: %.4f' % training_cost

        if vote_to_terminate >= 30:
            pass
            # print 'Training terminated due to no improvement!'
            # early_stopping = True
    print 'Best validation accuracy: %.4f' % best_accuracy

    print 'Training the network with all available data...'
    data = (
    np.concatenate((training_data[0], validation_data[0])), np.concatenate((training_data[1], validation_data[1])))
    # load_weights('pretrained/vgg16_weights.npz', model)

    # placeholder_lr.set_value(np.cast[theano.config.floatX](initial_learning_rate))
    for i in range(best_epoch):
        print 'Epoch %d starts...' % (i + 1)
        batches = utils.generator(data, net.batch_size)
        training_cost = 0.
        for idx, b in enumerate(batches):
            iteration = i * num_training_batches + idx + 1
            # utils.decrease_learning_rate(placeholder_lr, iteration, initial_learning_rate, final_learning_rate, 1000)

            x, y = b
            x = x.astype(theano.config.floatX) #if np.random.randint(0, 2) \
                # else seq.augment_images(x.astype(theano.config.floatX) * 255.) / 255.
            utils.update_input((x.transpose(0, 3, 1, 2), y), (placeholder_x, placeholder_y))
            training_cost += train(i + 1)
        training_cost /= num_training_batches
        print '\ttraining cost: %.4f' % training_cost
    # pkl.dump(model, open(save_file, 'wb'))
    # print 'Final model dumped to %s' % save_file
    print 'Training ended after %.2f hours' % ((time.time() - start_training_time) / 3600.)


if __name__ == '__main__':
    import read_data
    # sets = read_data.read_cifar10('C:\Users\just.anhduc\Downloads\cifar-10-batches-py', (32, 32), 0.2, True)
    # sets = [None, None]
    X_train, y_train, X_test, y_test = read_data.load_dataset('C:\Users\just.anhduc\Downloads')
    X_val, y_val = X_train[-5000:], y_train[-5000:]
    X_train, y_train = X_train[:-5000], y_train[:-5000]
    kwargs = {'training_data': (X_train, y_train), 'validation_data': (X_val, y_val)}
    train('densenet.config', **kwargs)
