import numpy as np

# credit: inspired by scikit-Learn make_datasets function, simplified for testing purposes
def generator_datasets(random_state=42, n_samples=10000, n_features=10, n_targets=1, n_informative=10, bias=0.02, noise=0.1):

    generator = np.random.RandomState(random_state)
    
    n_informative = min(n_features, n_informative)
    
    X = generator.standard_normal(size=(n_samples, n_features))
    
    beta = np.zeros((n_features, n_targets))
    beta[:n_informative, :] = 100 * generator.uniform(size=(n_informative, n_targets))
    
    Y = X @ beta + bias
    
    if noise > 0.0: Y += generator.normal(scale=noise, size=Y.shape)
    
    return X, Y.flatten(), beta.flatten()
