# -*- coding: utf-8 -*-
"""
=====================
Kriging Interpolation
=====================
Implemented by Juan Chacon @ UNESCO-IHE
Integrated Water Systems and Governance Department
Hydroinformatics Laboratory

This library simple and ordinary Kriging interpolation. Other Kriging \
applications such as universal kriging or universal Kriging, will be \
implemented in a posterior stage.

* Pre requisites
    you will need the following libraries, not coming alongside with the \ 
    Anaconda ditribution (recommended)

    * pyOpt - Optimisation engine, used to solve the semivariogram fitting  

* Functions
    * exp_semivariogram: Computes experimental semivariogram
    * theor_variogram: Adjust theoretical to experimental semivariogram
    * kriging_core: Performs data parsing and preprocessing for kriging \
    interpolation
    * krig: Solves the Kriging system and returns location-wise estimates of \
    points of interest
    * simple_Krig: Simple inteface for Kriging interpolation, by reading \
    standard csv files of data, and generating pkl of interpolation results
    * test: Test for Kriging module to see if everything is running as it \
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

import variogram_fit
import dist
import numpy as np
from numpy import linalg
import random
from pyOpt import ALHSO, Optimization
import data_save
import data_load
from time import ctime
from mat_corr import near_psd

ERROR_CODE = 9999
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
        if np.max(records[i]) > 3:
            WetMeas.append(records[i])    
    
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

def theor_variogram(experimental_sv, Sb=(0.01,400), Rb=(2,20), Nb=(0,400),
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
        candidate_sv = [variogram_fit.SVExponential, 
                        variogram_fit.SVGaussian,  
                        variogram_fit.SVSpherical, 
                        variogram_fit.SVCubic,
                        variogram_fit.SVPentaspherical, 
                        variogram_fit.SVSinehole, 
                        variogram_fit.SVPower]
     
    if candidate_sv_tag is None:
    # Names of functions for display only
        candidate_sv_tag = ['Exponential','Gaussian','Spherical','Cubic',
                     'Pentaspherical','Sinehole','Power','Matern']
    
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

def _kriging_core(ModOpt, single_target, stations, candidate_sv, xopt, 
                  record_covariance_matrix, records, krig_type):
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
            print('Non-singular covriance matrix - Sorry, cannot invert \n')
            err_out = ERROR_CODE*np.ones(len(records)) 
            return err_out, err_out

        InvSVr = linalg.inv(SVr)

        Z = []        
        WM= np.dot(InvSVr,SVm)
        for i in xrange(0,len(records)):
            Ztemp = np.dot(WM[:-1], records[i])
            #Ztemp = np.clip(Ztemp, 0, max(records[i])) # cutoff at 0 and max prec
            Z.append(Ztemp)        
            
        S = SVm[:-1]
        SP = xopt[0] - (np.dot(WM[:-1], np.transpose(S))) - WM[-1]
        
    elif krig_type is 'Sim':  # Simple Kriging
    
        SVr = np.array(record_covariance_matrix)        
        if linalg.det(record_covariance_matrix) == 0:
            print('Non-singular covriance matrix - Sorry, cannot invert \n')
            err_out = ERROR_CODE*np.ones(len(records))
            return err_out, err_out
        
        InvSVr = linalg.inv(SVr)        
        Z = []        
        #WM= np.dot(InvSVr, np.transpose(np.array(SVm)))    
        WM= np.dot(InvSVr, SVm)    
        for i in xrange(len(records)):
            Ztemp = np.dot(WM, records[i])
            #Ztemp = np.clip(Ztemp, 0, max(records[i])) # cutoff at 0 and max prec
            Z.append(Ztemp)        
        #S = xopt[0]*np.ones(len(SVm)) - SVm
        S = SVm
        SP = xopt[0] - (np.dot(WM, np.transpose(SVm)))
    
    else:
        print 'I pity the fool for no chosing Kriging type'
        print 'only available Ord and Sim \n'
        Z = ERROR_CODE*np.ones(len(records))
        SP = ERROR_CODE*np.ones(len(records))
    
    return Z, SP

def krig(MaxDist, targets, stations, records, record_covariance_matrix, ModOpt, 
         xopt, candidate_sv, tmin=0, tmax='def', MinNumSt='def', 
         krig_type='Sim', normalisation=False):
    '''
    Parsing of data for Kriging interpolation. This function contains\
    execution of interpolation as well. \n
    Parameters
    ----------
        **MaxDist** -- Initial search radius for nearby stations \n
        **targets** -- Points of interest to interpolate (targets) ``[t,2]`` \n
        **stations** -- Gauge location for interpolation ``[x,2]`` \n
        **records** -- Precipitation register for gauges ``[n,x]`` \n
        **record_covariance_matrix** -- Gauge records covariance matrix ``[x,x]`` \n
        **ModOpt** -- pointer to optimal semivariogram model \n
        **xopt** -- vector with optimal semivariogram parameters ``[5]`` \n
        **candidate_sv** -- Array with pointer to functions in variogram_fit module
            \n
        **tmin** -- Initial time step to be interpolated \n
        **tmax** -- Final time step to be interpolated \n
        **MinNumSt** -- Minimum number of stations within the search radius \n
        **krig_type** -- Type of Kriging to be used. 'Sim' for Simple and 'Ord'\
            for Ordinary Kriging
        **normalisation** -- Boolean. If true, then the variable is normalised\
             in the neighbourhood of the variable
            
    
    Returns
    -------
        **Z** -- Interpolation for each target and time step ``[n,t]`` \n
        **SP** -- Interpolation variance field ``[t,1]``\n
        **ZAvg** -- Average of interpolated field ``[n,1]``
    '''
    if MinNumSt is 'def':
        MinNumSt = len(stations)
        
    if MinNumSt > len(stations):
        print('Number of stations should be larger than number of \
               minimum stations. Set to all stations')
        MinNumSt = len(stations)
    
    if tmax == 'def':
        tmax = len(records)
    tmin = int(tmin)
    tmax = int(tmax)
    PrecSec = records[tmin:tmax]
    
    # Reduce measurements to relevant locations for the targets
    Z = []
    SP = []
    for kk in xrange(len(targets)):
        # Check if there are enough stations for interpolation, otherwise,
        # increase search radius
        targets_dt = dist.target(stations, [targets[kk]])[0]
        TNS = 0
        MaxDist2 = MaxDist
        # TODO fix this loop to smartly look for the next station
#        while TNS <= MinNumSt:
#            #Minimum distance
#            selected_stations = [i for i, v in enumerate(targets_dt) 
#                                if v > MaxDist2]
#            TNS = len(stations) - len(selected_stations)
#            MaxDist2 += 1.0
        
        das = np.array(dist.target([targets[kk],], stations)).flatten()
        
        #Select index of the closest stations
        cs = list(das.argsort()[:MinNumSt])
        selected_stations = stations[cs]
        
        # Reduction of relevant stations (reduced data and cov matrices)            
        RedLoc = np.delete(stations,selected_stations, 0)
        reduced_records = np.delete(PrecSec, selected_stations, 1)
        
        if normalisation:
            # detrending at all steps, one location
            local_average = np.average(reduced_records, 1).reshape(
                                                    [len(reduced_records), 1])
            reduced_records = reduced_records - local_average
            
        reduced_cov_matrix = record_covariance_matrix[:]
        reduced_cov_matrix = np.delete(reduced_cov_matrix, 
                                       selected_stations, 0)
        reduced_cov_matrix = np.delete(reduced_cov_matrix, 
                                       selected_stations, 1)
        
        # Kriging interpolation
        TempRes = _kriging_core(ModOpt, targets[kk], RedLoc, candidate_sv, 
                                xopt, reduced_cov_matrix, reduced_records, 
                                krig_type)
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
    SP[SP < 0] = 0
    return Z, SP, ZAvg

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
                    candidate_tag=['Spherical',]):
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
            print ('Variogram fitted for interval {0} at {1}'.format(interval, 
                                                                     ctime()))
        
        x_opt_st.append(x_opt_db)
        mod_opt_st.append(mod_opt_db)
    
    x_opt_st = np.array(x_opt_st)
    mod_opt_st = np.array(mod_opt_st)
    return x_opt_st, mod_opt_st


def ns_kriging_d(data, stations, bp, x_opt_st, hyper_sv, mod_opt_st, targets,
               candidate_var=[variogram_fit.spherical_sv,],
               candidate_tag=['Spherical',], verbose=False, par_map=None, 
               ncs=3, only_par_map=False):
    '''
    t1_var : list
        Parameters of the tier 1 variogram (variogram of the kriging 
        parameters) to be interpolated at each interval
        
    Tihs uses a spatial distribution of the kriging parameters and conditions
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
        
        sill_int, _, _ = krig(MaxDist=5.0, targets=targets,
                                      stations=stations, records=sill_map,
                                      record_covariance_matrix=hyper_cov_matrix,
                                      ModOpt=0, xopt=hyper_sv,
                                      candidate_sv=[variogram_fit.spherical_sv,],
                                      MinNumSt=ncs, normalisation=True, 
                                      krig_type='Ord')
        
        range_int, _, _ = krig(MaxDist=5.0, targets=targets,
                                      stations=stations, records=range_map,
                                      record_covariance_matrix=hyper_cov_matrix,
                                      ModOpt=0, xopt=hyper_sv,
                                      candidate_sv=[variogram_fit.spherical_sv,],
                                      MinNumSt=ncs, normalisation=True
                                      , krig_type='Ord')
        
        nug_int, _, _ = krig(MaxDist=5.0, targets=targets,
                                      stations=stations, records=nug_map,
                                      record_covariance_matrix=hyper_cov_matrix,
                                      ModOpt=0, xopt=hyper_sv,
                                      candidate_sv=[variogram_fit.spherical_sv,], 
                                      MinNumSt=ncs, normalisation=True, 
                                      krig_type='Ord')
        
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
            z, sp, _ = krig(MaxDist=5.0, targets=[targets[t],], 
                                 stations=stations, records=[data[n, :],], 
                                 record_covariance_matrix=temp_cov_mat, 
                                 ModOpt=0, xopt=temp_var, 
                                 candidate_sv=candidate_var, 
                                 MinNumSt=ncs, krig_type='Ord', 
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
    t1_var : list
        Parameters of the tier 1 variogram (variogram of the kriging 
        parameters) to be interpolated at each interval
        
    the method uses Ordinary Kriging using multi-precipitation model
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
                
    
                # Construct Covariance Matrix between stations 
                # for each precipitation event                
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


def test():
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
    
    Z, SP, ZAvg = krig(10.0, targets, stations, records, 
                       record_covariance_matrix, ModOpt, xopt, candidate_sv, 
                       10 ,20, 'Ord')
    print 'Ordinary Kriging working fine'
    
    Z, SP, ZAvg = krig(10.0, targets, stations, records, 
                       record_covariance_matrix, ModOpt, xopt, candidate_sv, 
                       10 , 20, 'Sim')
    print 'Simple Kriging working fine'
    return 'Ended normally, module should be working properly'