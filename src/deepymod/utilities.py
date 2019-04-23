import numpy as np
import os
import tensorflow as tf
import pandas as pd
import sys

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


def tensorboard_to_dataframe(event_path):
    data_iterator = tf.train.summary_iterator(event_path)

    data = []
    epoch = []
    tags = []
    while True:
        try:
            event = data_iterator.__next__()
            data.append([value.simple_value for value in event.summary.value])
            epoch.append(event.step)
            if (event.step > 0) and (len(tags) == 0):
                tags = [value.tag for value in event.summary.value]
        except:
            break
    data = np.array(data[3:]) #first three steps contain bullshit
    epoch = np.array(epoch[3:])

    # Parsing data into a nice dataframe
    idx_coeffs = []
    idx_coeffs_scaled = []

    for idx, term in enumerate(tags):
        if term[:5] == 'Coeff':
            idx_coeffs.append(idx)
        
        elif term[:6] == 'Scaled':
            idx_coeffs_scaled.append(idx)
        
    coeffs = np.take(data, idx_coeffs, axis=1)
    coeffs_scaled = np.take(data, idx_coeffs_scaled, axis=1)
    
    df = pd.DataFrame({'coeffs': list(coeffs),'coeffs_scaled': list(coeffs_scaled), 'epoch': epoch})
  
    for tag_idx in np.arange(5):
        df[str(tags[tag_idx])] = data[:, tag_idx]
        
    return df


def sparse_vec_classifier(test_vec, correct_vec):
    non_zero_terms_test = np.nonzero(test_vec)[0]
    non_zero_terms_correct = np.nonzero(correct_vec)[0]

    if np.array_equiv(non_zero_terms_test, non_zero_terms_correct) == 1:
        return 1, np.nanmean(np.abs((test_vec - correct_vec)/test_vec))*100
    else:
        return 0, None
