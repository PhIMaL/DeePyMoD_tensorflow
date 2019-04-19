import numpy as np
import tensorflow as tf
import sys
import os

sys.path.append('src/DeepMoD')
from graphs import PINN_graph, inference_graph
from tb_setup import tb_setup

def PINN(data, target, mask, config, library_function, library_config, train_opts, output_opts):
    # Defining graph, optimizer and feed_dict
    graph = PINN_graph(config, library_function, library_config)

    train_op = tf.train.AdamOptimizer(learning_rate=train_opts['learning_rate'], beta1=train_opts['beta1'], beta2=train_opts['beta2'], epsilon=train_opts['epsilon']).minimize(graph.loss)

    feed_dict = {graph.data_feed: data, graph.target_feed: target, graph.mask_feed: mask}

    # Running the fitting procedure
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)
        sess.run(graph.iterator.initializer, feed_dict=feed_dict)

        writer = tf.summary.FileWriter(os.path.join(output_opts['output_directory'], "iteration_" + str(output_opts['cycles'])))
        writer.add_graph(sess.graph)
        merged_summary, custom_board = tb_setup(graph, output_opts)
        writer.add_summary(custom_board)

        print('Epoch | Total loss | Loss gradient | MSE | PI | L1 ')
        for iteration in np.arange(train_opts['max_iterations']):
            sess.run(train_op)
            if iteration % 50 == 0:
                summary = sess.run(merged_summary)
                writer.add_summary(summary, iteration)
            if iteration % 500 == 0:
                print(iteration, sess.run([graph.loss, graph.gradloss, graph.cost_MSE, graph.cost_PI, graph.cost_L1]))
                if sess.run(graph.gradloss) < train_opts['grad_tol']:
                    print('Optimizer converged.')
                    break
          

        coeff_list = [map_to_sparse_vector(coeff_mask, coeff) for coeff_mask, coeff in zip(np.split(mask, mask.shape[1], axis=1), sess.run(graph.coeff_list))]
        coeff_scaled_list = [map_to_sparse_vector(coeff_mask, coeff) for coeff_mask, coeff in zip(np.split(mask, mask.shape[1], axis=1), sess.run(graph.coeff_scaled_list))]

        weights_biases = sess.run(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))[len(coeff_list):]
        weights = weights_biases[::2]
        biases = weights_biases[1::2]

    return coeff_list, coeff_scaled_list, weights, biases

def inference(data, weights, biases, layers, batchsize=1000):
    graph = inference_graph(data, weights, biases, layers, batchsize=batchsize)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
    
        prediction = [sess.run(graph.prediction) for batch in np.arange(np.ceil(data.shape[0]/batchsize))]
        prediction = np.concatenate(prediction, axis=0)

    return prediction


def map_to_sparse_vector(mask, coeff):
    sparse_vec = np.zeros_like(mask, dtype=np.float)
    sparse_vec[np.where(mask)[0]] = coeff

    return sparse_vec
