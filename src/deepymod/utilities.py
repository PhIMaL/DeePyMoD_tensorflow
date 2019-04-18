import numpy as np
import os
import tensorflow as tf
from re import search
import pandas as pd


def library_matrix_mat(u, v, latex=False):
    '''
    Implements the matrix multiplication for strings and flattens it,
    mimicking how the library is made.
    Set latex=True to obtain latex forms.
    '''
    comp_list = []
    for u_element in u:
        for v_element in v:
            if ((u_element == '1') and ('v_element' == '1')):
                result = '1'
            elif u_element == '1':
                result = v_element
            elif v_element == '1':
                result = u_element
            else:
                result = u_element + v_element
            comp_list.append(result)
    if latex is True:
        comp_list = list(map(lambda x: '$'+x+'$', comp_list))
    return comp_list


def print_PDE(sparse_vector, coeffs_list, PDE_term='u_t'):
    '''
    Prints PDE with non-zero components according to sparse_vector.
    Set PDE_term to different string for different equations.
    '''
    non_zero_idx = np.nonzero(sparse_vector)[0]
    PDE = PDE_term + ' = '
    for idx, term in enumerate(non_zero_idx):
        if idx != 0:
            if np.sign(sparse_vector[term]) == -1:
                PDE += ' - '
            else:
                PDE += ' + '
        PDE += '%.3f' % np.abs(sparse_vector[term]) + coeffs_list[term]
    print(PDE)


def tb_to_npy(event_folder):
    '''
    Returns tensorboard event file in event_folder as npy dict.
    '''
    event_path = os.path.join(event_folder, os.listdir(event_folder)[0])
    data = []
    # parsing event data
    for event in tf.train.summary_iterator(event_path):
        data.append([value.simple_value for value in event.summary.value])
        if event.step == 0:
            tags = [value.tag for value in event.summary.value]
    data = np.array(data[3:])

    # establishing search pattern for vectors
    coeffs_pattern = []
    coeffs_scaled_pattern = []
    scaling_pattern = []
    for idx, term in enumerate(tags):
        if search(r'Unscaled_*', term) is not None:
            coeffs_pattern.append(idx)

        if search(r'Scaled_*', term) is not None:
            coeffs_scaled_pattern.append(idx)

        if search(r'Scaling_*', term) is not None:
            scaling_pattern.append(idx)

    # Putting everything into a nice dictionary
    coeffs = np.take(data, coeffs_pattern, axis=1)
    coeffs_scaled = np.take(data, coeffs_scaled_pattern, axis=1)
    scaling = np.take(data, scaling_pattern, axis=1)

    data_dict = {}
    for tag_idx in np.arange(6):
        data_dict.update({str(tags[tag_idx]): data[:, tag_idx]})
        data_dict.update({'coeffs': coeffs,
                          'coeffs_scaled': coeffs_scaled,
                          'scaling': scaling})

    return data_dict


def tb_to_dataframe(event_folder):
    '''
    Returns tensorboard event file in event_folder as pandas dataframe.
    '''
    event_path = os.path.join(event_folder, os.listdir(event_folder)[0])
    data = []
    # parsing event data
    for event in tf.train.summary_iterator(event_path):
        data.append([value.simple_value for value in event.summary.value])
        if event.step == 0:
            tags = [value.tag for value in event.summary.value]
    data = np.array(data[3:])

    # establishing search pattern for vectors
    coeffs_pattern = []
    coeffs_scaled_pattern = []
    scaling_pattern = []
    for idx, term in enumerate(tags):
        if search(r'Unscaled_*', term) is not None:
            coeffs_pattern.append(idx)

        if search(r'Scaled_*', term) is not None:
            coeffs_scaled_pattern.append(idx)

        if search(r'Scaling_*', term) is not None:
            scaling_pattern.append(idx)

    # Putting everything into a nice dictionary
    coeffs = np.take(data, coeffs_pattern, axis=1)
    coeffs_scaled = np.take(data, coeffs_scaled_pattern, axis=1)
    scaling = np.take(data, scaling_pattern, axis=1)

    data_dict = {}
    for tag_idx in np.arange(6):
        data_dict.update({str(tags[tag_idx]): data[:, tag_idx]})

    data_dict.update({'coeffs': coeffs.tolist(),
                      'coeffs_scaled': coeffs_scaled.tolist(),
                      'scaling': scaling.tolist()})

    return pd.DataFrame(data_dict)


def sparse_vec_classifier(test_vec, correct_vec):
    non_zero_terms_test = np.nonzero(test_vec)[0]
    non_zero_terms_correct = np.nonzero(correct_vec)[0]

    if np.array_equiv(non_zero_terms_test, non_zero_terms_correct) == 1:
        return 1, np.nanmean(np.abs((test_vec - correct_vec)/test_vec))*100
    else:
        return 0, None
