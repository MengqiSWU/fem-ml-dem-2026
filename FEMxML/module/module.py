import tensorflow.compat.v1 as tf
# import tensorflow as tf
tf.disable_v2_behavior()


def get_states(model, processed_input, initial_hidden):
    all_hidden_states = tf.scan(model, processed_input, initializer=initial_hidden, name='states')
    all_hidden_states = all_hidden_states[:, 0, :, :]
    return all_hidden_states


def get_output(Wo, bo, hidden_state):
    output = tf.nn.relu(tf.matmul(hidden_state, Wo) + bo)
    return output


class LSTM_cell(object):
    def __init__(self, input_nodes, hidden_unit, output_nodes):

        self.input_nodes = input_nodes
        self.hidden_unit = hidden_unit
        self.output_nodes = output_nodes

        self.Wi = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))
        self.Ui = tf.Variable(tf.zeros([self.hidden_unit, self.hidden_unit]))
        self.bi = tf.Variable(tf.zeros([self.hidden_unit]))

        self.Wf = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))
        self.Uf = tf.Variable(tf.zeros([self.hidden_unit, self.hidden_unit]))
        self.bf = tf.Variable(tf.zeros([self.hidden_unit]))

        self.Wog = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))
        self.Uog = tf.Variable(tf.zeros([self.hidden_unit, self.hidden_unit]))
        self.bog = tf.Variable(tf.zeros([self.hidden_unit]))

        self.Wc = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))
        self.Uc = tf.Variable(tf.zeros([self.hidden_unit, self.hidden_unit]))
        self.bc = tf.Variable(tf.zeros([self.hidden_unit]))
        # weights for hidden input
        self.Wh = tf.Variable(tf.zeros([self.output_nodes, self.hidden_unit]))

        # Weights for output layers
        self.Wo = tf.Variable(tf.truncated_normal([self.hidden_unit, self.output_nodes], mean=0, stddev=.01))
        self.bo = tf.Variable(tf.truncated_normal([self.output_nodes], mean=0, stddev=.01))

        # Placeholder for input vector with shape[batch, seq, embeddings]
        self.hidden_input = tf.placeholder(tf.float32, shape=[None, self.output_nodes], name='y_hidden')
        self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.input_nodes], name='x_placeholder')
        # print('placeholder_xxxx')

        self.hidden_input_temp = tf.matmul(self.hidden_input, self.Wh)
        self.initial_hidden_temp = tf.matmul(self.inputs[:, 0, :], tf.zeros([input_nodes, hidden_unit]))

        self.initial_hidden = tf.stack([self.hidden_input_temp, self.initial_hidden_temp])

    def Lstm(self, previous_hidden_memory_tuple, x):
        # Take previous hidden stats and memory tuple with i/p &
        # o/p current hidden state
        previous_hidden_state, c_prev = tf.unstack(previous_hidden_memory_tuple)

        i = tf.sigmoid(tf.matmul(x, self.Wi) +
                       tf.matmul(previous_hidden_state, self.Ui) + self.bi)    # shape=(timestep, hidden_unit)

        f = tf.sigmoid(tf.matmul(x, self.Wf) +
                       tf.matmul(previous_hidden_state, self.Uf) + self.bf)    # shape=(timestep, hidden_unit)

        o = tf.sigmoid(tf.matmul(x, self.Wog) +
                       tf.matmul(previous_hidden_state, self.Uog) + self.bog)  # shape=(timestep, hidden_unit)

        c_ = tf.nn.tanh(tf.matmul(x, self.Wc) +
                        tf.matmul(previous_hidden_state, self.Uc) + self.bc)   # shape=(timestep, hidden_unit)

        # Final Memory cell
        c = f * c_prev + i * c_
        current_hidden_state = o * tf.nn.tanh(c)

        ans = tf.stack([current_hidden_state, c])

        return ans

    def get_states(self):
        all_hidden_states = tf.scan(fn=self.Lstm, elems=tf.transpose(self.inputs, [1, 0, 2]),
                                    initializer=self.initial_hidden, name='states')
        all_cell_states = all_hidden_states[:, 1, :, :]
        all_hidden_states = all_hidden_states[:, 0, :, :]
        all_cell_states = tf.transpose(all_cell_states, [1, 0, 2])
        all_hidden_states = tf.transpose(all_hidden_states, [1, 0, 2])
        return all_hidden_states, all_cell_states

    def get_output(self, hidden_state):
        output = tf.nn.sigmoid(tf.matmul(hidden_state, self.Wo) + self.bo)
        return output

    def get_cell_stats(self, cell_state):
        # output = tf.matmul(cell_state, tf.Constant())
        return cell_state

    def get_outputs(self, name):
        all_hidden_states, all_cell_states = self.get_states()
        all_outputs = tf.map_fn(self.get_output,
                                all_hidden_states,
                                name=name)
        # all_cell_states = tf.map_fn(self.get_cell_stats,
        #                             all_cell_states,
        #                             name=name+'_cell_states')
        tf.add_to_collection(name, all_outputs)
        tf.add_to_collection(name+'_cell_states', all_cell_states)
        return all_outputs


class GRU_cell(object):

    def __init__(self, input_nodes, hidden_unit, output_nodes):

        self.input_nodes = input_nodes
        self.hidden_unit = hidden_unit
        self.output_nodes = output_nodes

        self.Wx = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))

        self.Wr = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))
        self.br = tf.Variable(tf.truncated_normal([self.hidden_unit], mean=1))
        
        self.Wz = tf.Variable(tf.zeros([self.input_nodes, self.hidden_unit]))
        self.bz = tf.Variable(tf.truncated_normal([self.hidden_unit], mean=1))

        self.Wh = tf.Variable(tf.zeros([self.hidden_unit, self.hidden_unit]))

        self.Wo = tf.Variable(tf.truncated_normal([self.hidden_unit, self.output_nodes], mean=1, stddev=.01))
        self.bo = tf.Variable(tf.truncated_normal([self.output_nodes], mean=1, stddev=.01))

        self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.input_nodes], name='inputs')

        # batch_input_ = tf.transpose(self.inputs, perm=[2, 0, 1])
        self.processed_input = tf.transpose(self.inputs, [1, 0, 2])

        self.initial_hidden = self.inputs[:, 0, :]
        self.initial_hidden = tf.matmul(self.initial_hidden, tf.zeros([input_nodes, hidden_unit]))

    def Gru(self, previous_hidden_state, x):

        z = tf.sigmoid(tf.matmul(x, self.Wz) + self.bz)
        r = tf.sigmoid(tf.matmul(x, self.Wr) + self.br)

        h_ = tf.tanh(tf.matmul(x, self.Wx) +
                     tf.matmul(previous_hidden_state, self.Wh) * r)

        current_hidden_state = tf.multiply( (1 - z), h_) + tf.multiply(previous_hidden_state, z)

        return current_hidden_state

    def get_states(self):
        all_hidden_states = tf.scan(fn=self.Gru,
                                    elems=self.processed_input,
                                    initializer=self.initial_hidden,
                                    name='states')

        all_hidden_states = tf.transpose(all_hidden_states, [1, 0, 2])
        return all_hidden_states

    def get_output(self, hidden_state):
        output = tf.nn.relu(tf.matmul(hidden_state, self.Wo) + self.bo)
        return output

    def get_outputs(self, name):
        all_hidden_states = self.get_states()
        all_outputs = tf.map_fn(fn=self.get_output,
                                elems=all_hidden_states,
                                name=name)
        tf.add_to_collection(name, all_outputs)
        return all_outputs
