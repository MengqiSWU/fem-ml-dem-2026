import tensorflow as tf
import random
import sys
import numpy as np
import os
import time
from utilSelf.general import mkdirsSelf

tf.compat.v1.disable_eager_execution()

try:
    from FEMxML.module.module import LSTM_cell
    from FEMxML.getSeriasData import getSeriasData, plot_prection, pickle_load, output_loss_history
except:
    from module.module import LSTM_cell
    from getSeriasData import getSeriasData, plot_prection, pickle_load, output_loss_history


def normalize3D(x, xMean, xStd):
    x_normed = (x[..., :, :] - xMean) / xStd
    return x_normed


def normalizeReverse3D(x_normed, xMean, xStd):
    x = x_normed[..., :, :] * xStd + xMean
    return x


def splitTrainValidation(strain, stress, tangent):
    train_size = int(len(strain) * 0.8)
    rnd_idx = np.random.permutation(len(strain))
    strain_train, strain_validation = strain[rnd_idx[:train_size]], strain[rnd_idx[train_size:]]
    stress_train, stress_validation = stress[rnd_idx[:train_size]], stress[rnd_idx[train_size:]]
    tangent_train, tangent_validation = tangent[rnd_idx[:train_size]], tangent[rnd_idx[train_size:]]
    return strain_train, strain_validation, stress_train, stress_validation, tangent_train, tangent_validation


class lstmNet():
    def __init__(self, strain, stress, tangent, numHidden, modelPath):
        self.strain, self.stress, self.tangent = strain, stress, tangent
        self.stress_placeholder_normed = tf.compat.v1.placeholder(tf.float32,
                                                                  shape=(None, None, len(self.stress[0, 0])),
                                                                  name='stress_placeholder_normed')
        self.tangent_placeholder_normed = tf.compat.v1.placeholder(tf.float32,
                                                                   shape=(None, None, len(self.tangent[0, 0])),
                                                                   name='tangent_placeholder_normed')
        self.output_placeholder_normed = tf.concat((self.stress_placeholder_normed, self.tangent_placeholder_normed),
                                                   axis=2)
        self.strain_mean, self.stress_mean, self.tangent_mean, self.strain_std, self.stress_std, self.tangent_std = \
            pickle_load('total_e_mean', 's_mean', 'tangent_mean', 'total_e_std', 's_std', 'tangent_std',
                        root_path='./')

        self.strain_normed = normalize3D(self.strain, self.strain_mean, self.strain_std)
        self.stress_normed = normalize3D(self.stress, self.stress_mean, self.stress_std)
        self.tangent_normed = normalize3D(self.tangent, self.tangent_mean, self.tangent_std)

        # split to training set and validation set
        self.strain_train_normed, self.strain_validation_normed, \
        self.stress_train_normed, self.stress_validation_normed, \
        self.tangent_train_normed, self.tangent_validation_normed = splitTrainValidation(
            self.strain_normed,
            self.stress_normed,
            self.tangent_normed)

        self.model = LSTM_cell(len(self.strain[0, 0]), numHidden, len(self.stress[0, 0]))

        self.y_output = self.model.get_outputs(name='y_output')

        self.loss = tf.reduce_sum(
            tf.square(self.y_output - self.stress_placeholder_normed), axis=None, name='loss')

        self.train_op = tf.compat.v1.train.AdamOptimizer().minimize(self.loss, name='train_op')

        self.modelPath = modelPath
        mkdirsSelf(self.modelPath)

    def train(self, patienceNum):
        model_saved_name = os.path.join(self.modelPath, r"my_model_stop.ckpt")
        # define the training method
        saver = tf.compat.v1.train.Saver()
        sess = tf.compat.v1.Session()
        sess.run(tf.compat.v1.global_variables_initializer())
        # init.run()
        start_time = time.time()
        total_batch = 0
        best_loss_val = 2 ** 32
        last_improved = 0
        loss_history = []
        epoch = 0
        while True:
            sess.run(self.train_op, feed_dict={
                self.model.hidden_input: self.stress_train_normed[:, 0, :],
                self.model.inputs: self.strain_train_normed[:, 1:, :],
                self.stress_placeholder_normed: self.stress_train_normed[:, 1:, :]
            })

            # validation every 10 epoch
            if epoch % 100 == 0:
                loss_train = sess.run(self.loss, feed_dict={
                    self.model.hidden_input: self.stress_train_normed[:, 0, :],
                    self.model.inputs: self.strain_train_normed[:, 1:, :],
                    self.stress_placeholder_normed: self.stress_train_normed[:, 1:, :]
                })
                loss_val = sess.run(self.loss, feed_dict={
                    self.model.hidden_input: self.stress_validation_normed[:, 0, :],
                    self.model.inputs: self.strain_validation_normed[:, 1:, :],
                    self.stress_placeholder_normed: self.stress_validation_normed[:, 1:, :]
                })

                stress_normed_validation = sess.run(self.y_output, feed_dict={
                    self.model.hidden_input: self.stress_validation_normed[:, 0, :],
                    self.model.inputs: self.strain_validation_normed[:, 1:, :],
                    self.stress_placeholder_normed: self.stress_validation_normed[:, 1:, :]
                })
                stress_validation = normalizeReverse3D(stress_normed_validation, self.stress_mean, self.stress_std)
                stress_validation_true = normalizeReverse3D(self.stress_validation_normed, self.stress_mean,
                                                            self.stress_std)
                stress_normed_train = sess.run(self.y_output, feed_dict={
                    self.model.hidden_input: self.stress_train_normed[:, 0, :],
                    self.model.inputs: self.strain_train_normed[:, 1:, :],
                    self.stress_placeholder_normed: self.stress_train_normed[:, 1:, :]
                })
                stress_train = normalizeReverse3D(stress_normed_train, self.stress_mean, self.stress_std)
                stress_train_true = normalizeReverse3D(self.stress_train_normed, self.stress_mean, self.stress_std)

                loss_history.append([epoch, loss_train, loss_val])
                # save the model when the validation loss decrease
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    last_improved = epoch
                    saver.save(sess, model_saved_name)
                    improved_str = 'improved!'
                else:
                    improved_str = ''

                # record the consumed time, and print 'improved ! ' when the validation loss decrease
                consumed_time = (time.time() - start_time) / 60.
                msg = 'Epoch:{0:>4}, Loss_Train: {1:>.6}, Loss_Val: {2:>.6}, Time: {3:>.1} mins {4}'
                print(msg.format(epoch, loss_train, loss_val, consumed_time, improved_str))

            # training termination while no improvement in self.patience steps
            if epoch - last_improved > patienceNum:
                print("No optimization for ", patienceNum, " steps, auto-stop in the ", " step!")
                break
            epoch += 1
        sess.close()
        output_loss_history(loss_history, path=self.modelPath, validation_flag=True)


if __name__ == "__main__":
    strain, strain_increment, stress, tangent, n = getSeriasData(path='/home/shguan/simu/DEM_2_4_result')
    lstmModel = lstmNet(strain=strain, stress=stress, tangent=tangent, numHidden=100, modelPath='./lstmModel')
    lstmModel.train(500)
