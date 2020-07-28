import numpy as np


def gen_problem(n_var, n_contrain):
    
    contrain = np.random.randint(low=-7, high=19, size=(n_var,n_contrain))
    bacis = np.eye(n_contrain)

    # A will contain the coefficients of the constraints 
    A = np.vstack((contrain,bacis)).T

    # b will contain the amount of resources 
    b = np.random.randint(low=-7, high=19, size=(n_contrain,))


    # c will contain coefficients of objective function Z
    cz = np.random.randint(low=-7, high=19, size=(n_var,))
    cb = np.zeros((n_contrain,))
    c = np.concatenate([cz,cb])
    
    return A, b, c