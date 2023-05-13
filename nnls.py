#
# Gradient-descent, anti-lopsided algorithm for non-negative least squares (NNLS).
#
# Nguyen, Duy Khuong, and Tu Bao Ho. "Anti-lopsided algorithm for
# large-scale nonnegative least square problems." 
# arXiv:1502.01645 (2015). https://arxiv.org/abs/1502.01645
# solveNQP implementation from https://github.com/edmundaberry/NNLS 
# 

import logging, gc
import numpy as np

def nnls_gdal(A, b, epsilon=1e-10, max_n_iter=1000, verbose=True):
    '''
    Gradient-descent, anti-lopsided algorithm for non-negative least
    squares NNLS.
    
    A: d, n real matix
    b: n real vector

    '''

    logger=None
    if verbose:
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger('nnls_gdal')
        
    # Covariance
    ATA = A.T.dot(A)
    ATb = A.T.dot(b)
                
    # Normalization factors
    H_diag = np.diag(ATA).copy().reshape((-1,1))
    Q_den  = np.sqrt(np.outer(H_diag, H_diag))
    Q_den[np.isclose(Q_den, 0, rtol=epsilon, atol=epsilon)] = 1.
    q_den  = np.sqrt(H_diag)
    q_den[np.isclose(q_den, 0, rtol=epsilon, atol=epsilon)] = 1.

    # Normalize
    Q =  ATA / Q_den
    q = -ATb.reshape(-1,1) / q_den
    
    del ATA, ATb, Q_den
    
    if logger is not None:
        logger.info(f'q min/max = {(np.min(q), np.max(q))}')
    
    # Solve NQP
    y, n_passive, grad_norm = solveNQP(Q, q, epsilon, max_n_iter, logger=logger)
    gc.collect()
    if logger is not None:
        logger.info(f'n_passive = {n_passive} grad_norm = {grad_norm}')
    
    # Undo normalization
    x = y / q_den

    
    # Return
    return x


def solveNQP(Q, q, epsilon, max_n_iter, logger=None):
    '''
    Solves the non-negative quadratic problem (NQP) with gradient
    method using Exact Line Search.

    Q: n, n real matrix
    q: n real vector
    '''

    n_rows, n_cols = Q.shape
    
    # Initialize
    x          = np.zeros((n_cols, 1), dtype=np.float32)
    x_diff     = np.zeros((n_cols, 1), dtype=np.float32)
    grad_f     = q.copy()
    grad_f_bar = q.copy()
    Q_dot_x_diff = np.zeros((n_cols, 1), dtype=np.float32)
    Q_dot_grad_f_bar = np.zeros((n_cols, 1), dtype=np.float32)
    passive_set = np.zeros((n_cols, 1), dtype=np.bool_)
    
    for i in range(max_n_iter):
    
        # Get passive set information
        np.logical_or(x > 0, grad_f - epsilon < 0, out=passive_set)
        n_passive = np.sum(passive_set)

        # Calculate gradient
        grad_f_bar[:] = grad_f - epsilon
        grad_f_bar[~passive_set] = 0
        grad_norm = np.vdot(grad_f_bar, grad_f_bar)

        if logger is not None and i % 10 == 0:
            logger.info(f'solveNQP: iteration {i}: n_passive = {n_passive} grad_norm = {grad_norm}')
        
        # Check if convergence condition met
        if (n_passive == 0 or (i > 0 and grad_norm < epsilon)):
            if logger is not None:
                logger.info(f'solveNQP: convergence at iteration {i}: n_passive = {n_passive} grad_norm = {grad_norm}')
            break
            
        # Exact line search
        np.dot(Q, grad_f_bar, out=Q_dot_grad_f_bar)
        alpha_den = np.vdot(grad_f_bar, Q_dot_grad_f_bar)
        if np.abs(alpha_den) > 0.:
            alpha = grad_norm / alpha_den
        else:
            alpha = grad_norm
            
        # Updates x
        x_diff[:] = x
        x_diff *= -1.
        x -= alpha * grad_f_bar
        np.maximum(x, 0., out=x)
        x_diff += x
    
        # Updates gradient
        np.dot(Q, x_diff, out=Q_dot_x_diff)
        grad_f += Q_dot_x_diff

        
    return x, n_passive, grad_norm


