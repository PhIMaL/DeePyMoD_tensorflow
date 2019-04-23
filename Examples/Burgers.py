# Imports
import numpy as np
from deepymod.DeepMoD import DeepMoD
from deepymod.library_functions import library_1D
from deepymod.utilities import library_matrix_mat, print_PDE

np.random.seed(42)  # setting seed for randomisation

# Loading data
data = np.load('Examples/data/burgers.npy', allow_pickle=True).item()
X = np.transpose((data['x'].flatten(), data['t'].flatten()))
y = np.real(data['u']).reshape((data['u'].size, 1))

noise_level = 0.05
number_of_samples = 1000

y_noisy = y + noise_level * np.std(y) * np.random.randn(y.size, 1)
idx = np.random.permutation(y.size)
X_train = X[idx, :][:number_of_samples]
y_train = y_noisy[idx, :][:number_of_samples]

u = ['1', 'u', 'uˆ2']
du = ['1', 'u_{x}', 'u_{xx}', 'u_{xxx}']
coeffs_list = library_matrix_mat(u, du)

# Configuring and running DeepMoD
config = {'layers': [2, 20, 20, 20, 20, 20, 1], 'lambda': 10e-6}
train_opts = {'max_iterations': 10000, 'grad_tol': 10**-6, 'learning_rate': 0.002, 'beta1': 0.99, 'beta2': 0.999, 'epsilon': 10**-8}
library_config = {'total_terms': len(coeffs_list), 'deriv_order': 3, 'poly_order': 2}
output_opts = {'output_directory': 'Examples/output/burgers/', 'X_predict': X}

sparse_vectors, denoised = DeepMoD(X_train, y_train, config, library_1D, library_config, train_opts, output_opts)

print('Inferred equation:')
print_PDE(sparse_vectors[0], coeffs_list, PDE_term='u_t')
