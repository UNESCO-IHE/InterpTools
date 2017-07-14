# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 14:54:28 2016

@author: chaco3
"""

import numpy as np

def near_psd(x, epsilon=0):
    '''
    Document source
    http://www.quarchome.org/correlationmatrix.pdf
    
    Parameters
    ----------
    x : array_like
      Covariance/correlation matrix
    epsilon : float
      Eigenvalue limit (usually set to zero to ensure positive definiteness)
      
    Returns
    -------
    near_cov : array_like
      closest positive definite covariance/correlation matrix
    '''
        
    try:
        if min(np.linalg.eigvals(x)) > epsilon:
            return x
    except:
        import numpy as np
        if min(np.linalg.eigvals(x)) > epsilon:
            return x
        
    # Removing scaling factor of covariance matrix
    n = x.shape[0]
    var_list = np.array([np.sqrt(x[i,i]) for i in xrange(n)])
    y = np.array([[x[i, j]/(var_list[i]*var_list[j]) for i in xrange(n)] for j in xrange(n)])
    
    # getting the nearest correlation matrix
    eigval, eigvec = np.linalg.eig(y)
    val = np.matrix(np.maximum(eigval, epsilon))
    vec = np.matrix(eigvec)
    T = 1/(np.multiply(vec, vec) * val.T)
    T = np.matrix(np.sqrt(np.diag(np.array(T).reshape((n)) )))
    B = T * vec * np.diag(np.array(np.sqrt(val)).reshape((n)))
    near_corr = B*B.T    
    
    # returning the scaling factors
    near_cov = np.array([[near_corr[i, j]*(var_list[i]*var_list[j]) for i in xrange(n)] for j in xrange(n)])
    return near_cov

if __name__ == '__main__':
    # This is a not positive-defined matrix
    a = np.array([[1.0, 0.9, 0.7], [0.9, 1.0, 0.3], [0.7, 0.3, 1.0]])
    #a = np.array([[1.0, 0.9, 0.7, 1], [0.9, 1.0, 0.3,1], [0.7, 0.3, 1.0,1], [1,1,1,0]])
    eig = np.linalg.eigvals(a)
    print(eig)
    print('A negative eigenvalue is not from a positive semi-definite matrix')
    
    b = near_psd(a)
    print('The transformed matrix is: ')
    print(b)
    print('and the transformed eigenvalues: ')
    print(np.linalg.eigvals(b))
    