# -*- coding: utf-8 -*-
"""
=====================
Kriging Interpolation
=====================
Implemented by Juan Chacon @ UNESCO-IHE
Integrated Water Systems and Governance Department
Hydroinformatics Laboratory

This library simple and ordinary Kriging interpolation. Other Kriging 
applications such as universal kriging or universal Kriging, will be 
implemented in a posterior stage.

From May 2017, it contains applications regarding non-stationary Kriging.
Pre requisites
--------------
    you will need the following libraries, not coming alongside with the
    Anaconda ditribution (recommended)

    pyOpt - Optimisation engine, used to solve the semivariogram fitting
    

Functions
---------
    exp_semivariogram: Computes experimental semivariogram
    theor_variogram: Adjust theoretical to experimental semivariogram
    kriging_core: Performs data parsing and preprocessing for kriging \
    interpolation
    krig: Solves the Kriging system and returns location-wise estimates of \
    points of interest
    simple_Krig: Simple inteface for Kriging interpolation, by reading \
    standard csv files of data, and generating pkl of interpolation results
    test: Test for Kriging module to see if everything is running as it \
    should

* Use policy
    * You should include the respective citation to the authors
    * If you find this tool usefull, you will give the main author a beer next\
    time you see him :)
    
* References
    * http://people.ku.edu/~gbohling/cpe940/Kriging.pdf
"""
#Libraries to be imported
#------------------------------------------------------------------------------
import pyximport
pyximport.install()

import sys
import os
sys.path.append(os.path.abspath('..\\Utilities'))
sys.path.append(os.path.abspath('..\\ErrMet'))

import variogram_fit
import dist
import numpy as np
from numpy import linalg
import random
from pyOpt import ALHSO, Optimization
import data_save
import data_load
from error_metrics import rmse
from time import ctime
from mat_corr import near_psd

ERROR_CODE = 9999
C_LIMIT = 30.0  # Limit for conditional number of covariance matrix
EPSILON = 0.01
#------------------------------------------------------------------------------
def _regul(lag, cov, min_bin, maxdist):
    '''
    Internal function that regularises data for semivariogram computation by \
    selecting a minimum number of bins or a maximum distance criteria for bin \
    size.
    
    Parameters
    ----------
    lag : array_like
        Variable holding the lags for regularisation 
    cov : array_like
        contains the data which is going to be averaged in the bin
    min_bin : int
        minimum number of bins of the output
    maxdist : float
        maximum lag distance of each bin
    
    Returns
    -------
    lag2 : array_like
        vector holding the regularised bin location
    cov2 : array_like
        vector holding the regularised average of the variable in the bin
    '''
    maxdif = np.max(lag) #always starting from 0
    num_bin_dist = (maxdif / maxdist)
    
    if num_bin_dist > min_bin:
        total_number_bins = num_bin_dist
    else:
        total_number_bins = min_bin

    dist_between_bins = maxdif / total_number_bins
    
    lag2 = []
    cov2 = []
    for i in xrange(0,int(total_number_bins)):
        #get indices of elements within the bin.
        indx = [k for k, x in enumerate(lag) if i*dist_between_bins < x <= 
                (i+1)*dist_between_bins]
        sumbin = 0        
        if indx != []:
            for j in xrange(0,len(indx)):
                sumbin += cov[indx[j]]  
            cov2.append(sumbin/len(indx))
            lag2.append(i*dist_between_bins + dist_between_bins/2)
    return lag2, cov2
    
def exp_semivariogram(records, stations):
    '''
    Public function for establishing the experimental semivariogram.
    
    Parameters
    ----------
    records : array_like, shape ``(p,n)``
        Vector for which semivariogram is going to be calculated.
    stations : array_like shape ``(n,2)``
        Vector with the `x,y` coordinates of the measurement stations.
    
    Returns
    -------
    experimental_sv : array_like, shape ``(n,n)``
        Experimental semivariogram vector composed of lag and semivariogram
    record_covariance_matrix : array_like, shape ``(n,n)``
        Covariance matrix for the recorded data. 
    '''
    ## Removal of no precipitation events   
    WetMeas = []
    for i in xrange(0,len(records)):
        if np.max(records[i]) > 0.5:
            WetMeas.append(records[i])    
#    WetMeas = records[:]
    
    ## Measurement covariance
    record_covariance_matrix = np.cov(np.transpose(WetMeas))
    Dis = dist.between(stations)
    
    ## Experimental Semivariogram
    experimental_sv = []
    for i in xrange(0,len(record_covariance_matrix)-1):
        for j in xrange(i+1,len(record_covariance_matrix)):
            Cov = record_covariance_matrix[i][j]
            Lag = Dis[i][j]
            experimental_sv.append([Lag, Cov])
            
    experimental_sv = np.array(experimental_sv)
    Lag2, Cov2 = _regul(experimental_sv[:,0],experimental_sv[:,1],15,3)
    experimental_sv = np.transpose(np.vstack((Lag2,Cov2)))
    return experimental_sv, record_covariance_matrix

def theor_variogram(experimental_sv, Sb=(0.01,400), Rb=(2,60), Nb=(0,400),
                    ab=(0,2), vb=(0,1000), candidate_sv=None,
                    candidate_sv_tag=None):
    '''
    Fitting of theoretical variogram
    Parameters
    ----------
        **experimental_sv** -- Experimental semivariogram ''[x,2]'', lag and \
            semivariogram \n
        **Sb** -- Boundaries on Sill of semivariogram ``(min,max)`` \n
        **Rb** -- Boundaries on Range of semivariogram ``(min,max)`` \n
        **Nb** -- Boundaries on Nugget of semivariogram ``(min,max)`` \n
        **ab** -- Boundaries on Power of semivariogram ``(min,max)`` (only \
            valid for power semivariogram) \n
        **vb** -- Boundaries on Shape parameter of semivariogram ``(min,max)``\
            (only valid for matérn type) \n
    
    Returns
    -------
        **xopt** -- Vector with optimal semivariogram parameters ``[5]`` \n
        **ModOpt** -- Pointer to optimal vector location \n
        **candidate_sv** -- Array with pointer to functions in variogram_fit \
        module
    '''                      
    
    if candidate_sv is None:
        # Array with functions to be called from the Variograms library
        candidate_sv = [variogram_fit.exponential_sv, 
                        variogram_fit.gaussian_sv]
     
    if candidate_sv_tag is None:
    # Names of functions for display only
        candidate_sv_tag = ['Exponential','Gaussian']
    
    # Initial seed for variogram fit
        
    sr = random.uniform(Sb[0], Sb[1])
    rr = random.uniform(Rb[0], Rb[1])
    nr = random.uniform(Nb[0], Nb[1])
    ar = random.uniform(ab[0], ab[1])
    vr = random.uniform(vb[0], vb[1])
    
    Var = []
    Res = []
    Mdl = [] 
    
    # Wrapper of minimisation function (RMSE) for semivariogram fitting
    def _opt_fun(x,*args):
        F, g, fail = variogram_fit.fit_function(x, experimental_sv, 
                                                j,candidate_sv)
        if F == ERROR_CODE:
            fail = 1

        else:
            Var.append(x)
            Res.append(F)
            Mdl.append(j)
        return F, g, fail
    
    # Optimisation starts to minimise differences between experimental and 
    # theoretical semivariograms
    for j in xrange(0,len(candidate_sv)):   
        VarProb = Optimization('Variogram Fitting: ' + candidate_sv_tag[j], 
                               _opt_fun)
        VarProb.addObj('RMSE')
        VarProb.addVar('Sill', 'c', lower=Sb[0], upper=Sb[1], value=sr)
        VarProb.addVar('Range', 'c', lower=Rb[0], upper=Rb[1], value=rr)
        VarProb.addVar('Nugget', 'c', lower=Nb[0], upper=Nb[1], value=nr)
        VarProb.addVar('Exponent (a)', 'c', lower=ab[0], upper=ab[1], value=ar)
        VarProb.addVar('Rank (v)', 'c', lower=vb[0], upper=vb[1], value=vr)
        
        args = (experimental_sv, j, candidate_sv, Var, Res, Mdl)
        optmz = ALHSO()
        optmz.setOption('fileout',0)
        optmz(VarProb)

    # Get pointer to best semivariogram
    k = np.argmin(Res)
    xopt = Var[k]
    ModOpt = Mdl[k]
    
    return xopt, ModOpt, candidate_sv
    
def theor_variogram_ml(records, stations, MinNumSt=3, krig_type='Ord', 
                       normalisation=True, perf=rmse, 
                       ModOpt=0, Sb=(0.01,400), Rb=(2,20), 
                       Nb=(0,400), ab=(0,2), vb=(0,1000), candidate_sv=None,
                       candidate_sv_tag=None):
    '''
    Fitting of theoretical variogram using a maximum likelihood approach (RMSE)
    
    Parameters
    ----------
        **records** -- series for all stations\n
        **stations** -- localisation of stations\n
        **Sb** -- Boundaries on Sill of semivariogram ``(min,max)`` \n
        **Rb** -- Boundaries on Range of semivariogram ``(min,max)`` \n
        **Nb** -- Boundaries on Nugget of semivariogram ``(min,max)`` \n
        **ab** -- Boundaries on Power of semivariogram ``(min,max)`` (only \
            valid for power semivariogram) \n
        **vb** -- Boundaries on Shape parameter of semivariogram ``(min,max)``\
            (only valid for matérn type) \n
    
    Returns
    -------
        **xopt** -- Vector with optimal semivariogram parameters ``[5]`` \n
        **ModOpt** -- Pointer to optimal vector location \n
        **candidate_sv** -- Array with pointer to functions in variogram_fit \
        module
    '''                      
    
    if candidate_sv is None:
        # Array with functions to be called from the Variograms library
        candidate_sv = [variogram_fit.exponential_sv, ]
     
    if candidate_sv_tag is None:
    # Names of functions for display only
        candidate_sv_tag = ['Exponential',]
    
    # Initial seed for variogram fit
    _experimental_sv, _ = exp_semivariogram(records, stations)
    xopt_init, ModOpt_init, _ = theor_variogram(_experimental_sv, Sb, Rb, Nb, 
                                                ab, vb, candidate_sv, 
                                                candidate_sv_tag)
    
    sr = xopt_init[0]
    rr = xopt_init[1]
    nr = xopt_init[2]
    ar = xopt_init[3]
    vr = xopt_init[4]
    
    Var = []
    Res = []
    Mdl = [] 
    
    # Wrapper of minimisation function (RMSE) for semivariogram fitting
    def _opt_fun(x, *args):
        F, fail = cross_validation(stations, records, ModOpt, x, candidate_sv, 
                     MinNumSt, krig_type, normalisation, perf, True)
        g = []
        
        if F == ERROR_CODE:  # redundant
            fail = 1

        else:
            print('RMSE: {0}'.format(F))
            Var.append(x)
            Res.append(F)
            Mdl.append(j)
        return F, g, fail
    
    # Optimisation starts to minimise differences between experimental and 
    # theoretical semivariograms
    for j in xrange(len(candidate_sv)):   
        VarProb = Optimization('Variogram Fitting: ' + candidate_sv_tag[j], 
                               _opt_fun)
        VarProb.addObj('RMSE')
        VarProb.addVar('Sill', 'c', lower=Sb[0], upper=Sb[1], value=sr)
        VarProb.addVar('Range', 'c', lower=Rb[0], upper=Rb[1], value=rr)
        VarProb.addVar('Nugget', 'c', lower=Nb[0], upper=Nb[1], value=nr)
        VarProb.addVar('Exponent (a)', 'c', lower=ab[0], upper=ab[1], value=ar)
        VarProb.addVar('Rank (v)', 'c', lower=vb[0], upper=vb[1], value=vr)
        
        args = (j, candidate_sv, Var, Res, Mdl)
        optmz = ALHSO()
        optmz.setOption('fileout', 0)
        optmz(VarProb)

    # Get pointer to best semivariogram
    if Res == []:
        xopt = xopt_init
        ModOpt = ModOpt_init
        print('Initial values taken, no ML used for variogram calibration')
    else:
        k = np.argmin(Res)
        xopt = Var[k]
        ModOpt = Mdl[k]
    
    return xopt, ModOpt, candidate_sv

def _kriging_core(ModOpt, single_target, stations, candidate_sv, xopt, 
                  record_covariance_matrix, records, krig_type, meas_var):
    '''
    Kriging core where interpolating algorithms is taking place.
    
    Parameters
    ----------
        **ModOpt** -- Pointer to optimal semivariogram model \n
        **single_target** -- Single point of interest to interpolate (targets)\
            ``[t,2]`` \n
        **stations** -- Gauge location for interpolation ``[x,2]`` \n
        **candidate_sv** -- Array with pointer to functions in variogram_fit module
            \n
        **xopt** -- vector with optimal semivariogram parameters ``[5]`` \n
        **record_covariance_matrix** -- Gauge records covariance matrix ``[x,x]`` \n
        **records** -- Precipitation register for gauges ``[n,x]`` \n
        **krig_type** -- Type of Kriging to be used. 'Sim' for Simple and 'Ord'\
            for Ordinary Kriging
    
    Returns
    -------
        **Z** -- Interpolation for each target and time step ``[n,1]`` \n
        **SP** -- Interpolation variance field ``[1]``\n
    '''
    n_stations = len(stations)
    targetsD = dist.target(stations, [single_target])[0]
    SVm = []
    for j in xrange(len(stations)):
        SVm.append(candidate_sv[ModOpt](targetsD[j], xopt))
        
    # fix covariance matrix so not having negative eigenvalues
    record_covariance_matrix = near_psd(record_covariance_matrix)
    
    if krig_type is 'Ord':  #Ordinary Kriging
        
        record_covariance_matrix = np.row_stack((record_covariance_matrix,
                                    np.ones(len(record_covariance_matrix))))
        record_covariance_matrix = np.column_stack((record_covariance_matrix,
                                    np.ones(len(record_covariance_matrix))))
        record_covariance_matrix[-1,-1] = 0.0
        SVm.append(1.0)
        
        SVr = np.array(record_covariance_matrix)
        if linalg.det(record_covariance_matrix) == 0:
            print('Non-singular covriance matrix - Sorry, cannot invert')
            err_out = ERROR_CODE*np.ones(len(records)) 
            return err_out, err_out
        
        if np.max(meas_var) == 0:
            Z = []
            InvSVr = linalg.inv(SVr)
            WM= np.dot(InvSVr,SVm)
          
            for i in xrange(len(records)):               
                Ztemp = np.dot(WM[:-1], records[i])
                Z.append(Ztemp)        
                
            S = SVm[:-1]
            SP = (xopt[0] + xopt[2]) - (np.dot(WM[:-1], np.transpose(S))) - WM[-1]
            
        else:
            Z = []
            SP = []
            for i in xrange(len(records)):
                records_i = records[i]
                add_var_mat = np.zeros([n_stations + 1, n_stations + 1])
                
                # generate added measurement covariance  matrix
                add_var_mat[:-1, :-1] = np.array([[np.sqrt(meas_var[i, j]*meas_var[i, k]) 
                                               for j in xrange(n_stations)] 
                                               for k in xrange(n_stations)])
                
                add_var_vec = np.zeros([n_stations + 1])
                add_var_vec[:-1] = np.array([np.sqrt(meas_var[i, j]) 
                                             for j in xrange(n_stations)])

                #augmented variance matrix with measurement noise (trimmed)
                SVr_mod = np.clip(SVr - add_var_mat, 0, np.inf)
                
                # Augmented variance towards target (trimmed)
                SVm_mod = np.clip(SVm - add_var_vec, 0, np.inf)
                
                # Check for colinearity (index) of the solutions and remove the non-necessary
                coll_vars = [j for j in xrange(len(SVr_mod)-1) if np.max(SVr_mod[:, j]) == 0]
                
                if coll_vars != []:
                    # Remove from covariance matrix
                    SVr_mod = np.delete(SVr_mod, coll_vars, 0)
                    SVr_mod = np.delete(SVr_mod, coll_vars, 1)
                    
                    # remove from variance to vector
                    SVm_mod = np.delete(SVm_mod, coll_vars, 0)
                    
                    # remove from records used
                    records_i = np.delete(records_i, coll_vars, 0)
                    
                # Check for conditionally-ill matrix and add elements to the 
                # diagonal to reduce the conditional number to 50      
#                if np.min(np.linalg.eigvals(SVr_mod)) < EPSILON:
#                    SVr_mod = near_psd(SVr_mod, EPSILON)
#                
#                # If the measurements are independent, the weights are the same
#                off_diag_mat = [SVr_mod[:-1, :-1] 
#                                - np.diag(np.diag(SVr_mod[:-1,:-1]))]
                
#                if np.max(off_diag_mat) == 0:
#                    # if measurements are independent
#                    Z.append(np.average(records_i))
#                    SP.append(xopt[0])
#                
#                else:
                    # Get the weights
                InvSVr = linalg.inv(SVr_mod)
                WM= np.dot(InvSVr, SVm_mod)
                
                if np.round(np.sum(WM[:-1]), 3) != 1.0:
                    print('Weights not equal to 1: {0}'.format(WM[:-1]))
                ## Check if weights are negative. If so, eliminate and re-scale
                if np.min(WM) < 0:
                    
                    # Eliminate negative values
                    WM[WM < 0] = 0
                    com_weight = np.sum(WM)
                    # Re-scale the rest of the weights to sum 1
                    WM = np.array(WM)/np.sum(com_weight)
                    
                Ztemp = np.dot(WM[:-1], records_i)
                #Ztemp = np.clip(Ztemp, 0, max(records[i])) # cutoff at 0 and max prec
                Z.append(Ztemp)
                
                S = SVm_mod[:-1]
                SP_temp = (xopt[0] + xopt[2]) - (np.dot(WM[:-1], np.transpose(S))) - WM[-1]
                
                if SP_temp < 0:
                    print('Negative variance')
                    print('Lagrange_param: {0}'.format(WM[-1]))
                    print('variance to target: {0}'.format(S))
                    print('reduction in variance: {0}'.format(np.dot(WM[:-1], np.transpose(S))))
                    print('weights: {0}'.format(WM[:-1]))
                    print('')
                SP.append(SP_temp)
        
    elif krig_type is 'Sim':  # Simple Kriging
    
        SVr = np.array(record_covariance_matrix)        
        if linalg.det(record_covariance_matrix) == 0:
            print('Non-singular covriance matrix - Sorry, cannot invert \n')
            err_out = ERROR_CODE*np.ones(len(records))
            return err_out, err_out
        
        if np.max(meas_var) == 0:
            InvSVr = linalg.inv(SVr) 
            WM= np.dot(InvSVr, SVm)  
            Z = []        
            for i in xrange(len(records)):
                Ztemp = np.dot(WM, records[i])
                Z.append(Ztemp)        
            S = SVm
            SP = (xopt[0] + xopt[2]) - (np.dot(WM, np.transpose(SVm)))
            
        else:
            Z = []
            SP = []
            for i in xrange(len(records)):
                records_i = records[i]
                # generate added measurement covariance  matrix
                add_var_mat = np.array([[np.sqrt(meas_var[i, j]*meas_var[i, k]) 
                                               for j in xrange(n_stations)] 
                                               for k in xrange(n_stations)])
                
                add_var_vec = np.array([np.sqrt(meas_var[i, j]) for j in xrange(n_stations)])

                #augmented variance matrix with measurement noise (trimmed)
                SVr_mod = np.clip(SVr - add_var_mat, 0, np.inf)
                
                # Augmented variance towards target (trimmed)
                SVm_mod = np.clip(SVm - add_var_vec, 0, np.inf)

                # Check for collinearity of the solutions and remove the non-necessary
                coll_vars = [j for j in xrange(len(SVr_mod)) if np.max(SVr_mod[:, j]) == 0]
                
                if coll_vars != []:
                    # Remove from covariance matrix
                    SVr_mod = np.delete(SVr_mod, coll_vars, 0)
                    SVr_mod = np.delete(SVr_mod, coll_vars, 1)
                    
                    # remove from variance to vector
                    SVm_mod = np.delete(SVm_mod, coll_vars, 0)
                    
                    # remove from records used
                    records_i = np.delete(records_i, coll_vars, 0)
                
                # Check for conditionally-ill matrix and add elements to the 
                # diagonal to reduce the conditional number to 50      
#                if np.min(np.linalg.eigvals(SVr_mod)) < EPSILON:
#                    SVr_mod = near_psd(SVr_mod, EPSILON)
                
                # Get the weights
                InvSVr = linalg.inv(SVr_mod)
                WM = np.dot(InvSVr, SVm_mod)
                
#                if np.sum(WM[:-1]) != 1.0:
#                    print('sum of weights not equal to 1: {0}'.format(WM))
#                    print('Eigenvalues: {0}'.format(np.linalg.eigvals(SVr_mod)))
#                if np.abs(np.linalg.cond(SVr_mod)) > 50:
#                    print('Potentially ill conditioned matrix')
#                    print('Eigenvalues: {0}'.format(np.linalg.eigvals(SVr_mod)))
#                    print('Condition number: {0}'.format(np.linalg.cond(SVr_mod)))
#                    print('Weights: {0}'.format(WM))
#                    print('')
#                
#                if np.min(WM) < 0:
#                    print('Negative weights: {0}'.format(WM))
#                    print('')     
                
                ## Check if weights are negative. If so, eliminate and re-scale
                if np.min(WM) < 0:
                    com_weight = 1.0 - np.sum(WM)
                    # Eliminate negative values
                    WM[WM < 0] = 0
                    
                    # Re-scale the rest of the weights to sum 1
                    WM = np.array(WM)/np.sum(com_weight)
                    
                Ztemp = np.dot(WM, records_i)
                Z.append(Ztemp)        
                S = SVm_mod
                SP_temp = (xopt[0] + xopt[2]) - (np.dot(WM, np.transpose(SVm_mod)))
                
                if SP_temp <= 0:
#                    print('Negative variance in estimation: {0}'.format(SP_temp))
#                    print('Weights: {0}'.format(WM))
#                    print('Variance contribution: {0}'.format(SVm_mod))
#                    print('Approximated to the average of variance \
#                           contribution: {0}'.format(np.average(SVm_mod)))
#                    print('')
                    SP_temp = np.average(SVm_mod)
                SP.append(SP_temp)
    
    else:
        print 'I pity the fool for no chosing Kriging type'
        print 'only available Ord and Sim \n'
        Z = ERROR_CODE*np.ones(len(records))
        SP = ERROR_CODE*np.ones(len(records))
    
    return Z, SP

def krig(MaxDist, targets, stations, records, record_covariance_matrix, ModOpt, 
         xopt, candidate_sv, tmin=0, tmax=None, MinNumSt=None, 
         krig_type='Sim', normalisation=False, m_error=None, verbose=False):
    '''
    Parsing of data for Kriging interpolation. This function contains
    execution of interpolation as well.
    
    Parameters
    ----------
    MaxDist : float
        Initial search radius for nearby stations
    targets : array_like
        Points of interest to interpolate (targets) ``[t,2]``
    stations : array_like
        Gauge location for interpolation ``[x,2]``
    records : array_like
        Precipitation register for gauges ``[n,x]``
    record_covariance_matrix : array_like
        Gauge records covariance matrix ``[x,x]``
    ModOpt : array_like
        pointer to optimal semivariogram model
    xopt : array_like
        vector with optimal semivariogram parameters ``[5]``
    candidate_sv : array_like
        Array with pointer to functions in variogram_fit module
    tmin : int
        Initial time step to be interpolated
    tmax : int
        Final time step to be interpolated
    MinNumSt : int
        Minimum number of stations within the search radius
    krig_type : str
        Type of Kriging to be used. 'Sim' for Simple and 'Ord' for Ordinary 
        Kriging
    normalisation : Bool
        Boolean. If true, then the variable is normalised in the neighbourhood 
        of the variable
    m_error : array_like
        cotains the variance in the measurement error
            
    Returns
    -------
    Z : array_like
        Interpolation for each target and time step ``[n,t]``
    SP : array_like 
        Interpolation variance field ``[t,1]``
    ZAvg : array_like
        Average of interpolated field ``[n,1]``
    
    '''
    #print (MinNumSt)
    if MinNumSt is None:
        MinNumSt = len(stations)
    
    if MinNumSt >= len(stations):
        if verbose: 
            print('Number of stations should be larger than number of \
                   minimum stations. Set to all stations')
        MinNumSt = len(stations)
    
    if m_error is not None:
        if np.max(m_error) > xopt[0]:
            print('Measurement error larger than process variance. The \
                   system will fail')
    
    if tmax is None:
        tmax = len(records)
    tmin = int(tmin)
    tmax = int(tmax)
    PrecSec = records[tmin:tmax]
    
    if m_error is None:
        m_error = np.zeros([len(PrecSec), len(stations)])
    else:
        m_error = m_error[tmin:tmax]
    
    # Reduce measurements to relevant locations for the targets
    Z = []
    SP = []
    for kk in xrange(len(targets)):
        # Check if there are enough stations for interpolation, otherwise,
        # increase search radius
        #targets_dt = dist.target(stations, [targets[kk]])[0]
        #TNS = 0
        #MaxDist2 = MaxDist    
        das = np.array(dist.target([targets[kk],], stations)).flatten()
        
        #Select index of the closest stations
        cs = list(das.argsort()[:MinNumSt])
        fs = list(das.argsort()[MinNumSt:])
        selected_stations = stations[cs]
        
        
        # Reduction of relevant stations (reduced data and cov matrices)            
        RedLoc = np.delete(stations, fs, 0)
        reduced_records = np.delete(PrecSec, fs, 1)
        
        # reduction of the measurement error matrix
        meas_var = m_error[:, cs]
        
        if normalisation:
            # detrending at all steps, one location
            local_average = np.average(reduced_records, 1).reshape(
                                                    [len(reduced_records), 1])
            reduced_records = reduced_records - local_average
            
        reduced_cov_matrix = record_covariance_matrix[:]
        reduced_cov_matrix = np.delete(reduced_cov_matrix,
                                       fs, 0)
        reduced_cov_matrix = np.delete(reduced_cov_matrix, 
                                       fs, 1)
        
        # Kriging interpolation
        TempRes = _kriging_core(ModOpt, targets[kk], RedLoc, candidate_sv, 
                                xopt, reduced_cov_matrix, reduced_records, 
                                krig_type, meas_var)
        if Z == []:
            if normalisation:
                Z = np.vstack(TempRes[0]) + local_average
            else:
                Z = np.vstack(TempRes[0]) 
            
        else:
            if normalisation:
                temp = np.vstack(TempRes[0]) + local_average
            else:            
                temp = np.vstack(TempRes[0])
                
            Z = np.hstack((Z, temp))        
        SP.append(TempRes[1])
    ZAvg = np.average(Z, 1)
    
    SP = np.array(SP)
    #SP[SP < 0] = 0
    return Z, SP, ZAvg

def cross_validation(stations, records, ModOpt, xopt, candidate_sv, 
                     MinNumSt=None, krig_type='Sim', normalisation=True, 
                     perf=rmse, calibration=False):
    _fail = 0
    _rec_cov_mat = np.array([[candidate_sv[ModOpt](kk, xopt) for
                                kk in row_dist] for row_dist in
                                dist.between(stations)])
    _int_res = []
    for st_i in xrange(len(stations)):
        _st_used = range(len(stations))
        _st_used.pop(st_i)
        _rec_cov_mat_red = np.array([[_rec_cov_mat[i, j] for i in _st_used] for j in _st_used])
        _temp_res = krig(0, [stations[st_i, :],], 
                         stations[_st_used, :], records[:, _st_used], 
                         _rec_cov_mat_red, ModOpt, 
                         xopt, candidate_sv, 0, None, MinNumSt, krig_type, 
                         normalisation, None)[0]
        if np.any(_temp_res == ERROR_CODE*np.ones(len(records))):
            _fail = 1
            if calibration:
                return ERROR_CODE, _fail
        _int_res.append(np.array(_temp_res).flatten())
    
    performance = perf(np.array(_int_res).flatten(), records.flatten())
    
    return performance, _fail
                         

def simple_Krig(SiteInfo, XYTargets, DataRecord):
    '''
    Wrapper for Kriging interpolation of all data, and save in PKL format \n
    Parameters 
    ----------
        **SiteInfo** -- Path to file with gauge location \n
        **XYTargets** -- Path to file with interpolation target locations \n
        **DataRecord** -- Path to file with variable registries \n

    Returns
    -------
        **VarField** -- file with pickled variable field \n
        **VarUnc** -- file with pickled variance of estimated variable \n
        **AvgVar** -- file with pickled average of estimated field
    '''
    
    stations, targets, records = data_load.lcsv(SiteInfo, XYTargets, 
                                                DataRecord)
    experimental_sv, record_covariance_matrix = exp_semivariogram(records, 
                                                                  stations)
    xopt, ModOpt, candidate_sv = theor_variogram(experimental_sv)
    Z, SP, ZAvg = krig(xopt[0]/3.0, targets, stations, records, 
                       record_covariance_matrix, ModOpt, xopt,
                       candidate_sv, tmin=0, tmax='def', MinNumSt=3, 
                       krig_type='Sim')

    return Z, SP, ZAvg

def multi_variogram(data, stations, bp, 
                    candidate_var=[variogram_fit.spherical_sv,],
                    candidate_tag=['Spherical',], verbose=False):
    '''   
    Calculates the object with multiple semivariograms of the data at different
    breaking points
    
    Parameters:
    -----------
    data : nd_array
        Array containing the measurements all the measurements of the variable
        on sze '[n, m]'. n is the number of measurements, m the number of 
        stations        
    stations : nd_array
        Array containing the location of stations. The size of the array is of 
        '[m, 2]'
    bp : list
        List with the sections of the data to create the pools. Do not include
        values equal to the maximum or minimum of the data.
    
    returns:
    --------
    
    xopt : nd_array
        3 dimensional array consisting of station index, condition index and
        variogram parameters
        
    mod_opt_st : nd_array
        3 dimensional array consisting of station index, condition index and 
        optimal model output
    '''
    #Make n variograms as breaking points in the location of the sensors
    # bp has to in ascending order
    bp = bp[:]
    bp.insert(0, np.min(data))
    bp.append(np.max(data))
    intervals = [[bp[i], bp[i+1]] for i in xrange(len(bp)-1)]
    intervals[-1][1] = intervals[-1][1]+0.0001
    
    x_opt_st = []
    mod_opt_st = []    
    
    # iterate over the stations
    for s_idx in xrange(len(stations)):
        # gest distance of station towards the other stations
        dist_to_stations = dist.target(stations, [stations[s_idx], ])
        
        # Iterate over the intervals of precipitation
        x_opt_db = []
        mod_opt_db = []
        for interval in intervals:
            # Go throgh data to get variogram for each condition at each station
            # Make the pool for the SV            
            temp_pool = []            
            for i in xrange(len(data)):
                if data[i, s_idx] >= min(interval) and \
                   data[i, s_idx] < max(interval):
                    temp_pool.append(data[i, :])
            
            # Raise error if pool is empty 
            if temp_pool == []:
                raise NameError('Pool is empty: no data at station {0} \
                on interval {1}. Pick new breaking points'.format(s_idx, 
                                                                  interval))            
            temp_pool = np.array(temp_pool)

            # get experimental variogram for the pool
            temp_cov_matrix = np.cov(np.transpose(temp_pool))
            reg_interval, reg_cov = _regul(np.transpose(dist_to_stations), 
                                                   temp_cov_matrix[:,s_idx], 
                                                    15, 1)
            
            temp_exp_sv = np.transpose(np.vstack((reg_interval, reg_cov)))
            
            # fit to get theoretical SV
            x_opt, mod_opt, _ = theor_variogram(temp_exp_sv,
                                              candidate_sv=candidate_var,
                                              candidate_sv_tag=candidate_tag,
                                              Sb = (0.01, 6.0),
                                              Rb = (5.00, 45.0),
                                              Nb = (0.001, 0.05))
            # save results
            x_opt_db.append(x_opt)
            mod_opt_db.append(mod_opt)
            if verbose: print ('Variogram fitted for interval {0} \
                                at {1}'.format(interval, ctime()))
        
        x_opt_st.append(x_opt_db)
        mod_opt_st.append(mod_opt_db)
    
    x_opt_st = np.array(x_opt_st)
    mod_opt_st = np.array(mod_opt_st)
    return x_opt_st, mod_opt_st


def ns_kriging_d(data, stations, bp, x_opt_st, hyper_sv, mod_opt_st, targets,
               candidate_var=[variogram_fit.spherical_sv,],
               candidate_tag=['Spherical',], 
               verbose=False, par_map=None, 
               ncs=3, only_par_map=False, krig_type='Ord'):
    '''
    Parsing of data for Kriging interpolation. This function contains
    execution of interpolation as well.
    
    Parameters
    ----------
    data : array_like
        Measurements for interpolation ``[n, x]``
    stations : array_like
        Gauge location for interpolation ``[x,2]``
    bp : array_like
        Precipitaiton value for creating a new variogram class
    xopt_st : array_like
        Array containing the vectors with optimal semivariogram parameters. 
        Use the output of the multivariogram function
        ``[5]``
    hyper_sv : array_like
        Parameters of the hyper (second layer) semivariogram to interpolate 
        the paramters along 
    mod_opt_st : array_like
        pointer to optimal semivariogram models. Use the output of the 
        multivariogram function
    targets : array_like
        Points of interest to interpolate (targets) ``[t,2]``
    candidate_var : array_like, optional
        Array with pointer to functions in variogram_fit module. Default is 
        spherical variogram
    candidate_tag : list, optional
        list with the tags of the used variograms. Default is spherical 
        variogram
    verbose : Bool, optional
        Print messages regarding the interpolation process. Default is False
    par_map : array_like, optional
        Map with the interpolated values of the Kriging parameters in the 
        position of the targets. Default, calculates this map internally
    ncs : int, optional
        Number of closest stations for interpolation. Default is 3.
    only_par_map : Bool, optional
        Early termination to only build the parameter maps. No interpolation 
        is made if selected as True.
    krig_type : str, optional
        Select the type of Kriging, 'Ord' for ordinary and 'Sim' for simple 
        Kriging.
            
    Returns
    -------
    z_n : array_like
        Interpolation for each target and time step ``[n,t]``
    sp_n : array_like 
        Interpolation variance field ``[t,1]``
    par_map : array_like
        Interpolation of the parameter set in the targets for each condition 
        ``[3, n, t]``
    '''
    
    if ncs > len(stations):
        print('Reduced number of minimum stations to total number of stations')
        ncs = len(stations)
        
    _bp = bp[:]
    _bp.insert(0, np.min(data))
    _bp.append(np.max(data))
    intervals = [[_bp[i], _bp[i+1]] for i in xrange(len(_bp)-1)]
    intervals[-1][1] = intervals[-1][1]+0.0001
    
    # Get the paramter maps in each of the intervals
    # Parameter maps have the shape of t targets, n samples
    if par_map is None:
        sill_map = np.zeros_like(data)
        range_map = np.zeros_like(data)
        nug_map = np.zeros_like(data)
        for st_i in xrange(len(stations)):
            for n in xrange(len(data)):
                for i in xrange(len(intervals)):
                    if data[n, st_i] >= min(intervals[i]) and \
                       data[n, st_i] < max(intervals[i]):
                        sill_map[n, st_i] = x_opt_st[st_i, i, 0]
                        range_map[n, st_i] = x_opt_st[st_i, i, 1]
                        nug_map[n, st_i] = x_opt_st[st_i, i, 2]
       
        # Make interpolation of the kriging parameters to the domain
        hyper_cov_matrix = np.array([[variogram_fit.spherical_sv(kk, hyper_sv) for
                                        kk in row_dist] for row_dist in
                                        dist.between(stations)])
        
        sill_int, _, _ = krig(MaxDist=5.0, 
                              targets=targets,
                              stations=stations, 
                              records=sill_map,
                              record_covariance_matrix=hyper_cov_matrix,
                              ModOpt=0, 
                              xopt=hyper_sv,
                              candidate_sv=[variogram_fit.spherical_sv,],
                              MinNumSt=ncs, 
                              normalisation=True, 
                              krig_type=krig_type)
        
        range_int, _, _ = krig(MaxDist=5.0, 
                               targets=targets,
                               stations=stations, 
                               records=range_map,
                               record_covariance_matrix=hyper_cov_matrix,
                               ModOpt=0, 
                               xopt=hyper_sv,
                               candidate_sv=[variogram_fit.spherical_sv,],
                               MinNumSt=ncs, 
                               normalisation=True,
                               krig_type=krig_type)
        
        nug_int, _, _ = krig(MaxDist=5.0, 
                             targets=targets,
                             stations=stations, 
                             records=nug_map,
                             record_covariance_matrix=hyper_cov_matrix,
                             ModOpt=0, 
                             xopt=hyper_sv,
                             candidate_sv=[variogram_fit.spherical_sv,], 
                             MinNumSt=ncs, 
                             normalisation=True, 
                             krig_type=krig_type)
        
        par_map = [sill_int, range_int, nug_int]
        if verbose: print ('Parameters interpolated in domain')
        if only_par_map: return par_map
    
    # do the interpolation
    z_n = []
    sp_n = []
    
    for n in xrange(len(data)):
        z_t = []
        sp_t = []

        for t in xrange(len(targets)):
            # Get the variogram parameters for the interpolation centres
            temp_var = [par_map[0][n, t], par_map[1][n, t], par_map[2][n, t], 0, 0]
    
            # Do the covariance matrix
            temp_cov_mat = np.array([[candidate_var[0](kk, temp_var) for 
                            kk in row_dist] for row_dist in dist.between(stations)])
            
            # Do the interpolation            
            z, sp, _ = krig(MaxDist=5.0, 
                            targets=[targets[t],], 
                            stations=stations, 
                            records=[data[n, :],], 
                            record_covariance_matrix=temp_cov_mat, 
                            ModOpt=0, 
                            xopt=temp_var, 
                            candidate_sv=candidate_var, 
                            MinNumSt=ncs, 
                            krig_type=krig_type, 
                            normalisation=True)
            
            # append results for each target
            z = np.max([z[0][0], 0])
            sp = np.max(sp[0], 0)
            
            z_t.append(z)
            sp_t.append(sp)

        if verbose: print ('Finished step {0}/{1} at {2}'.format(n, len(data), 
                                                                 ctime()))
        z_n.append(z_t)
        sp_n.append(sp_t)
    
    z_n = np.array(z_n)
    sp_n = np.array(sp_n)
    return z_n, sp_n, par_map

def ns_kriging_mm(data, stations, bp, x_opt_st, mod_opt_st, targets,
               candidate_var=[variogram_fit.spherical_sv,],
               candidate_tag=['Spherical',], verbose=False, 
               ncs=3, krig_type='Ord'):
    '''
    Make the interpolation using the Average  bi-directional covariance matrix 
    approach.
    
    Parameters
    ----------
    data : array_like
        Measurements for interpolation ``[n, x]``
    stations : array_like
        Gauge location for interpolation ``[x,2]``
    bp : array_like
        Precipitaiton value for creating a new variogram class
    xopt_st : array_like
        Array containing the vectors with optimal semivariogram parameters. 
        Use the output of the multivariogram function
        ``[5]``
    hyper_sv : array_like
        Parameters of the hyper (second layer) semivariogram to interpolate 
        the paramters along 
    mod_opt_st : array_like
        pointer to optimal semivariogram models. Use the output of the 
        multivariogram function
    targets : array_like
        Points of interest to interpolate (targets) ``[t,2]``
    candidate_var : array_like, optional
        Array with pointer to functions in variogram_fit module. Default is 
        spherical variogram
    candidate_tag : list, optional
        list with the tags of the used variograms. Default is spherical 
        variogram
    verbose : Bool, optional
        Print messages regarding the interpolation process. Default is False
    par_map : array_like, optional
        Map with the interpolated values of the Kriging parameters in the 
        position of the targets. Default, calculates this map internally
    ncs : int, optional
        Number of closest stations for interpolation. Default is 3.
    only_par_map : Bool, optional
        Early termination to only build the parameter maps. No interpolation 
        is made if selected as True.
    krig_type : str, optional
        Select the type of Kriging, 'Ord' for ordinary and 'Sim' for simple 
        Kriging.
            
    Returns
    -------
    z_n : array_like
        Interpolation for each target and time step ``[n,t]``
    sp_n : array_like 
        Interpolation variance field ``[t,1]``
    '''
    if krig_type is 'Ord':
        ord_krig=True
        
    # Set the intervals
    bp = bp[:]
    bp.insert(0, np.min(data))
    bp.append(np.max(data))
    intervals = [[bp[i], bp[i+1]] for i in xrange(len(bp)-1)]
    intervals[-1][1] = intervals[-1][1]+0.0001
    
    # Make separation of the regime for each record
    cond_map = np.zeros_like(data)
    for st_i in xrange(len(stations)):
        for n in xrange(len(data)):
            for i in xrange(len(intervals)):
                if data[n, st_i] >= min(intervals[i]) and \
                   data[n, st_i] < max(intervals[i]):
                    cond_map[n, st_i] = i
    
    # Get distance between stations
    dist_mat = np.array(dist.between(stations))
    
    #initialisation of lists for each target
    zz = np.zeros([len(data), len(targets)])
    ssp = np.zeros([len(data), len(targets)])
    
    for tarcoun in xrange(len(targets)):
        
        #Select invidual target for itnerpolation
        tar = targets[tarcoun]
        
        ##Select the closest NCS elements
        #Calculate distance of all stations
        das = np.array(dist.target([tar,], stations)).flatten()
        
        #Select index of the closest stations
        cs = list(das.argsort()[:ncs])
        
        #Reduction of precipitation and sensor position matrices      
        red_data = data[:, cs]
        red_dist = dist_mat[cs][:,cs]
        red_cond_map = cond_map[:, cs]
        red_xopt = x_opt_st[cs]

        # Pre-cooked all zeros case
        if ord_krig:
            cov_mat_zeros = np.ones([ncs+1, ncs+1])
            cov_mat_zeros[-1, -1] = 0
        else:
            cov_mat_zeros = np.ones([ncs, ncs])
            
        for ii in xrange(ncs):
            for jj in xrange(ncs):
                # TODO include different variogram type where 0 is
                cov_mat_zeros[ii, jj] = candidate_var[0](red_dist[ii, jj], 
                                                         red_xopt[ii, 0])
        
        if ord_krig:
            cov_tar_zeros = np.ones([ncs+1, 1])
        else:
            cov_tar_zeros = np.ones([ncs, 1])
            
        for ii in xrange(ncs):
            cov_tar_zeros[ii, 0] = candidate_var[0](das[cs[ii]], 
                                                    red_xopt[ii, 0])
        
        # Inverse of extended covariance matrix
        inv_cov_mat_zeros = np.linalg.inv(cov_mat_zeros)
                
        # Kriging weights
        w_vec = np.dot(inv_cov_mat_zeros, cov_tar_zeros)
        
        if np.min(w_vec) < 0:
            if ord_krig:
                # Eliminate negative values
                w_vec[w_vec < 0] = 0
                com_weight = np.sum(w_vec)
                # Re-scale the rest of the weights to sum 1
                w_vec = np.array(w_vec)/np.sum(com_weight)
                
            else:
                com_weight = 1.0 - np.sum(w_vec)
                # Eliminate negative values
                w_vec[w_vec < 0] = 0
                
                # Re-scale the rest of the weights to sum 1
                w_vec = np.array(w_vec)/np.sum(com_weight)

        
        # Interpolation variance and Kriging system solution
        if ord_krig:
            sp_zeros = (np.max(cov_mat_zeros) 
                     - (np.sum(w_vec * cov_tar_zeros + w_vec[-1][0])))
        else:
            sp_zeros = (np.max(cov_mat_zeros) 
                     - (np.dot(np.transpose(w_vec), cov_tar_zeros)[0][0]))
        
        for t in xrange(len(data)):
            
            # If there is some precipitation data
            if np.max(red_data[t, :]) != 0:
                ## Here the interpolation is made for each of the time steps at 
                ## each locatiom
                #Make covariance matrix between stations in the neighborhoud (K)
                if ord_krig:
                    cov_mat = np.ones([ncs+1, ncs+1])
                    cov_mat[-1, -1] = 0
                else:
                    cov_mat = np.ones([ncs, ncs])
                
                # Construct Covariance Matrix between stations for each 
                # precipitation event                
                for ii in xrange(ncs):
                    for jj in xrange(ncs):
                        # TODO include different variogram type where 0 is
                        cov_mat[ii, jj] = candidate_var[0](red_dist[ii, jj], 
                                        red_xopt[ii, int(red_cond_map[t, ii])])
                
                # Fix maximum diagonal element to maximum of the variogram
                # To force positive eigenvalues
                for ii in xrange(ncs):
                    cov_mat[ii, ii] = np.max(cov_mat)
                
                # Simmetrise using the average bi-directional covariance matrix
                cov_mat = 0.5*(cov_mat + np.transpose(cov_mat))
                
                # Construct covariance towards target
                if ord_krig:
                    cov_tar = np.ones([ncs+1, 1])
                else:
                    cov_tar = np.ones([ncs, 1])
                    
                for ii in xrange(ncs):
                    cov_tar[ii, 0] = candidate_var[0](das[cs[ii]], 
                                     red_xopt[ii, int(red_cond_map[t, ii])])
                
                ## Solve the Kriging system
                # Invert the covariance matrix
                inv_cov_mat = np.linalg.inv(cov_mat)
                
                # Kriging weights
                w_vec = np.dot(inv_cov_mat, cov_tar)
                
                if np.min(w_vec) < 0:
                    if ord_krig:
                        # Eliminate negative values
                        w_vec[w_vec < 0] = 0
                        com_weight = np.sum(w_vec)
                        # Re-scale the rest of the weights to sum 1
                        w_vec = np.array(w_vec)/np.sum(com_weight)
                        
                    else:
                        com_weight = 1.0 - np.sum(w_vec)
                        # Eliminate negative values
                        w_vec[w_vec < 0] = 0
                        
                        # Re-scale the rest of the weights to sum 1
                        w_vec = np.array(w_vec)/np.sum(com_weight)
                
                # Interpolation variance and Kriging system solution
                if ord_krig:
                    z_temp = np.dot(np.transpose(w_vec[:-1]), 
                                    np.reshape(red_data[t, :], [-1,1]))[0][0]
                    #sp_temp = (np.max(cov_mat) 
                    #           - (np.sum(w_vec * cov_tar + w_vec[-1][0])))
                    sp_temp = (np.max(cov_mat) 
                               - (np.sum(w_vec * cov_tar) + w_vec[-1][0]))
                else:
                    z_temp = np.dot(np.transpose(w_vec), 
                                    np.reshape(red_data[t, :], [-1,1]))[0][0]
                    sp_temp = np.max(cov_mat) - (np.sum(w_vec * cov_tar))

                # Appending of solutions
                ssp[t, tarcoun] = sp_temp
                zz[t, tarcoun] = z_temp
                                
            else:
                ssp[t, tarcoun] = sp_zeros
                zz[t, tarcoun] = 0
            
            if verbose: 
                print ('Finished step {0}/{1} at {2}'.format(n, len(data), 
                                                                 ctime()))

    return zz, ssp

if __name__ == '__main__':
    
    '''
    Module testing function
    '''
    stations, targets, records = data_load.lcsv('TestData\GaugeLoc.csv',
                                       'TestData\InterpPts.csv',
                                       'TestData\Dataset.csv')
    stations = np.array(stations)/1000.0
    targets = np.array(targets)/1000.0
    experimental_sv, record_covariance_matrix = exp_semivariogram(records, 
                                                                  stations)
    xopt, ModOpt, candidate_sv = theor_variogram(experimental_sv)
    
    meas_err_mat = np.random.rand(len(records), len(stations))    
    
    Z, SP, ZAvg = krig(10.0, targets, stations, records, 
                       record_covariance_matrix, ModOpt, xopt, candidate_sv, 
                       10 ,20, MinNumSt=3, krig_type='Ord')
    print 'Ordinary Kriging working fine'
    
    Z, SP, ZAvg = krig(10.0, targets, stations, records, 
                       record_covariance_matrix, ModOpt, xopt, candidate_sv, 
                       10 , 20, MinNumSt=3, krig_type='Sim')
    print 'Simple Kriging working fine'
    print 'Ended normally, module should be working properly'
    
    Z, SP, ZAvg = krig(10.0, targets, stations, records, 
                       record_covariance_matrix, ModOpt, xopt, candidate_sv, 
                       10 ,20, MinNumSt=3, krig_type='Ord', m_error=meas_err_mat)
    print 'Ordinary Kriging working fine'
    
    Z, SP, ZAvg = krig(10.0, targets, stations, records, 
                       record_covariance_matrix, ModOpt, xopt, candidate_sv, 
                       10 , 20, MinNumSt=3, krig_type='Ord', m_error=meas_err_mat)
    print 'Simple Kriging with error working fine'
    print 'Ended normally, module with error should be working properly'
