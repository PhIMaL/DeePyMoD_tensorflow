import numpy as np
import tensorflow as tf


def PINN_graph(config, library_function, library_config):
    tf.reset_default_graph()
    # Creating datasets
    with tf.name_scope("Defining_variables"):
        data_feed = tf.placeholder(tf.float32, shape=[None, config['layers'][0]])
        target_feed = tf.placeholder(tf.float32, shape=[None, config['layers'][-1]])
        mask_feed = tf.placeholder(tf.int32, shape=[None, config['layers'][-1]])
        BC_mask = tf.placeholder(tf.int32, shape=[None, 1])

        lambda_L1 = tf.constant(config['lambda'], tf.float32)

        coeff_list = [tf.Variable(config['initial_coeffs'][output_neuron], dtype=tf.float32) for output_neuron in np.arange(config['layers'][-1])]
        BC_value = tf.Variable(1.0, dtype=tf.float32)

    with tf.name_scope("Data_pipeline"):
        mask = tf.ones([tf.size(target_feed[:, 0:1]), tf.shape(mask_feed)[0], tf.shape(mask_feed)[1]], dtype=tf.int32) * tf.expand_dims(mask_feed, axis=0)

        dataset = tf.data.Dataset.from_tensor_slices((data_feed, target_feed, mask)).repeat().batch(tf.shape(data_feed, out_type=tf.int64)[0])

        iterator = dataset.make_initializable_iterator()
        data, target, sparsity_mask = iterator.get_next()

    # The actual network
    with tf.name_scope("Neural_Network"):
        X = data
        for layer in np.arange(len(config['layers'])-2):
            X = tf.layers.dense(X, units=config['layers'][layer+1], activation=tf.nn.tanh, kernel_initializer=tf.constant_initializer(config['initial_weights'][layer]), bias_initializer=tf.constant_initializer(config['initial_biases'][layer]))
        prediction = tf.layers.dense(inputs=X, units=config['layers'][-1], activation=None, kernel_initializer=tf.constant_initializer(config['initial_weights'][-1]), bias_initializer=tf.constant_initializer(config['initial_biases'][-1]))

    # make library according to supplied function
    with tf.name_scope("Creating_library"):
        time_deriv_list, theta = library_function(data, prediction, library_config)
        theta_split = [tf.dynamic_partition(theta, coeff_mask, 2)[1] for coeff_mask in tf.unstack(sparsity_mask, axis=2, num=len(coeff_list))]
        sparse_thetas_list = [tf.reshape(sparse_theta, [tf.shape(theta)[0], tf.size(coeff)]) for coeff, sparse_theta in zip(coeff_list, theta_split)]

    # Normalizing
    with tf.name_scope("Scaling"):
        scaling_time = [tf.norm(time_deriv, axis=0) for time_deriv in time_deriv_list]
        scaling_theta = [tf.expand_dims(tf.norm(sparse_theta, axis=0), axis=1) for sparse_theta in sparse_thetas_list]
        coeff_scaled_list = [coeff * (theta_scale / time_scale) for coeff, theta_scale, time_scale in zip(coeff_list, scaling_theta, scaling_time)]

    # Defining cost function
    with tf.name_scope("Cost_MSE"):
        MSE_costs = tf.reduce_mean(tf.square(target - prediction), axis=0)
        cost_MSE = tf.reduce_mean(MSE_costs)

    with tf.name_scope("Cost_PI"):
        PI_costs = [tf.reduce_mean(tf.square(tf.matmul(sparse_theta, coeff) - time_deriv)) for sparse_theta, coeff, time_deriv in zip(sparse_thetas_list, coeff_list, time_deriv_list)]
        cost_PI = tf.reduce_sum(PI_costs)

    with tf.name_scope('Cost_L1'):
        L1_costs = [lambda_L1 * tf.reduce_sum(tf.abs(coeff[1:, :])) for coeff in coeff_scaled_list]
        cost_L1 = tf.reduce_sum(L1_costs)

    with tf.name_scope('Cost_BC'):
        #bc_set = tf.gather_nd(theta[:, 3], BC_mask)  # Dirichlet
        bc_set = tf.gather_nd(theta[:, 1], BC_mask)  # Neumann
        cost_BC = tf.reduce_mean(tf.square(bc_set - BC_value))

    with tf.name_scope("Total_cost"):
        loss = cost_MSE + cost_PI + cost_L1 + cost_BC

    # graph node for gradient
    with tf.name_scope("GradLoss"):
        grad_losses = [tf.reduce_max(tf.abs(tf.gradients(loss, coeff)[0]) / (theta_scale / time_scale)) for coeff, theta_scale, time_scale in zip(coeff_list, scaling_theta, scaling_time)]
        gradloss = tf.reduce_max(grad_losses)

    return AttrDict(locals())


def inference_graph(data, weights, biases, layers, batchsize=1000):
    dataset = tf.data.Dataset.from_tensor_slices(data).batch(batchsize)
    iterator = dataset.make_one_shot_iterator()
    data = iterator.get_next()

    X = data
    for layer in np.arange(len(layers)-2):
        X = tf.layers.dense(X, units=layers[layer+1], activation=tf.nn.tanh, kernel_initializer=tf.constant_initializer(weights[layer]), bias_initializer=tf.constant_initializer(biases[layer]))
    prediction = tf.layers.dense(inputs=X, units=layers[-1], activation=None, kernel_initializer=tf.constant_initializer(weights[-1]), bias_initializer=tf.constant_initializer(biases[-1]))

    return AttrDict(locals())


class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
