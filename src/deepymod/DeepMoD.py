import numpy as np
import copy

from deepymod.PINN import PINN, map_to_sparse_vector, inference

def DeepMoD(data, target, config, library_function, library_config, train_opts, output_opts):
    # Defining internal configuration
    internal_config = copy.deepcopy(config)

    # Defining initial weights, biases and coefficients for the network
    initial_coeffs = [np.random.rand(library_config['total_terms'], 1) * 2 - 1 for output_neuron in np.arange(config['layers'][-1])]
    initial_biases = [np.zeros(neurons) for neurons in config['layers'][1:]]
    initial_weights = [np.random.randn(input_neurons, output_neurons) * np.sqrt(1 / (input_neurons + output_neurons)) for input_neurons, output_neurons in zip(config['layers'][:-1], config['layers'][1:])]  # Xavier initalization
    internal_config.update({'initial_coeffs': initial_coeffs, 'initial_weights': initial_weights, 'initial_biases': initial_biases})

    # Run minimization procedure
    mask = np.ones((library_config['total_terms'], config['layers'][-1]))
    converged = False
    output_opts.update({'cycles': 0})

    while not converged:
        coeff_list, coeff_scaled_list, weights, biases = PINN(data, target, mask, internal_config, library_function, library_config, train_opts, output_opts)
        sparsity_pattern_list = [thresholding(coeff, mode='auto') for coeff in coeff_scaled_list]

        # Checking convergence before doing anything else
        output_opts['cycles'] += 1
        converged = convergence(sparsity_pattern_list, mask)

        # Updating everything else for next cycle
        mask[~np.transpose(np.squeeze(np.array(sparsity_pattern_list)))] = 0
        coeff_list_thresholded = [np.expand_dims(coeff[sparsity_pattern], axis=1) for coeff, sparsity_pattern in zip(coeff_list, sparsity_pattern_list)]
        internal_config.update({'initial_coeffs': coeff_list_thresholded, 'initial_weights': weights, 'initial_biases': biases})
        

        # Printing current sparse vector to see progress
        print('Current sparse vectors:')
        print([map_to_sparse_vector(sparsity_pattern, coeff) for sparsity_pattern, coeff in zip(sparsity_pattern_list, coeff_list_thresholded)])

    # Now thats it's converged, fit again but without the L1 penalty
    print('Now running for the final time...')
    internal_config['lambda'] = 0
    coeff_list, _, weights, biases = PINN(data, target, mask, internal_config, library_function, library_config, train_opts, output_opts)

    if 'X_predict' in output_opts.keys():
        prediction = inference(output_opts['X_predict'], weights, biases, internal_config['layers'])
        return coeff_list, prediction
    else:
        return coeff_list


def thresholding(vector, mode, treshold=0.0):
    if mode == 'auto':
        upper_lim, lower_lim = np.median(vector)+np.std(vector), np.median(vector) - np.std(vector)
        sparsity_mask = (vector <= upper_lim) & (vector >= lower_lim)
    else:
        sparsity_mask = np.abs(vector) < treshold

    return ~sparsity_mask


def convergence(coeff_list, mask):
    coeff_active_terms = np.array([np.count_nonzero(coeff) for coeff in coeff_list])
    mask_active_terms = np.count_nonzero(mask, axis=0)

    return np.array_equal(coeff_active_terms, mask_active_terms)
