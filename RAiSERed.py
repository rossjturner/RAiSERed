# RAiSERed module
# Ross Turner, 17 May 2020

# import packages
import astropy.units as u
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches

import numpy as np
import scipy.fftpack
import warnings
from astropy.cosmology import FlatLambdaCDM
from functools import partial
from matplotlib import rc
from multiprocessing import cpu_count, Pool
from scipy.stats import chi2, skewnorm
from noisyopt import minimizeCompass

# define RAiSERed data structures
class RAiSERed_list(list):

    def __init__(self, *args):
        if args == ():
            super(RAiSERed_list, self).__init__([])
        elif isinstance(args[0], RAiSERed_obj):
            super(RAiSERed_list, self).__init__([args[0]])
        else:
            raise Exception('List input must be a RAiSERed object.')

    def __getitem__(self, index):
        if not isinstance(index, str):
            return super(RAiSERed_list, self).__getitem__(index)
        else:
            for element in self:
                if element.source == index:
                    return element
            raise Exception('List index \''+str(index)+'\' is not an element of the RAiSERed_list.')
        
class RAiSERed_obj:

    def __init__(self, source, freq, flux, size, axis, inj, brkfreq, specz, calib, mean, stdev, x, y, raw, series, nredshifts, nsimulations):
        # source name
        self.source = source
        # radio continuum measurements
        self.freq = freq
        self.flux = flux
        self.size = size
        self.axis = axis
        self.inj = inj
        self.brkfreq = brkfreq
        # spectroscopic redshift
        self.specz = specz
        self.calib = calib
        # probability distributions
        self.mean = mean
        self.stdev = stdev
        self.x = x
        self.y = y
        self.raw = raw
        # additional variables
        self.series = series
        self.nredshifts = nredshifts
        self.nsimulations = nsimulations


#### RAiSERed_add function and sub-functions ####
# define basic data manipulation functions
def RAiSERed_add(raisered_list, source, freq, flux, size, axis, inj, brkfreq, specz = None, calib = True, series = None, nredshifts = 50, nsimulations = 10000):
    
    # set warnings to run as expected
    warnings.filterwarnings('always')
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # use source name in error messages if provided
    if not isinstance(source, str):
        raise Exception('Source name must be of type string.')
    else:
        error_str = source+' - '

    # verify minimum number of redshifts and simulations selected
    if isinstance(nredshifts, (int, float)):
        nredshifts = max(25, int(nredshifts))
    else:
        raise Exception(error_str+'Number of redshifts must be of type integer.')
    if isinstance(nsimulations, (int, float)):
        nsimulations = max(5000, int(nsimulations))
    else:
        raise Exception(error_str+'Number of simulations must be of type integer.')

    # calculate skewed normal probability distribution for each input variable
    frequency_x = __variable_distribution(error_str+'Frequency', freq, 10*nsimulations)
    flux_density_x = __variable_distribution(error_str+'Flux density', flux, 10*nsimulations)
    angular_size_x = __variable_distribution(error_str+'Angular size', size, 10*nsimulations)
    break_freq_x = __variable_distribution(error_str+'Break frquency', brkfreq, 10*nsimulations)
    inject_index_x = __variable_distribution(error_str+'Injection index', inj, 10*nsimulations)
    axis_ratio_x = __variable_distribution('Axis ratio', axis, 10*nsimulations)
        
    # check values are within accepted ranges
    if np.any(frequency_x <= 0):
        raise Exception(error_str+'Frequency in at least one simulation is < 0 Hertz.')
    elif np.all(frequency_x < 20):
        frequency_x = frequency_x*1e9
        warnings.warn(error_str+'Frequency has been converted from Gigahertz into the expected unit of Hertz.', category=UserWarning)
    elif np.all(frequency_x < 1e5):
        frequency_x = frequency_x*1e6
        warnings.warn(error_str+'Frequency has been converted from Megahertz into the expected unit of Hertz.', category=UserWarning)
    if np.any(flux_density_x <= 0):
        raise Exception(error_str+'Flux density in at least one simulation is < 0 Janskys.')
    if np.any(angular_size_x <= 0):
        warnings.warn(error_str+'Angular size in at least one simulation is < 0 arcseconds.')
    if np.all(break_freq_x > 100):
        break_freq_x = np.log10(break_freq_x)
        warnings.warn(error_str+'Break frequency has been converted from Hertz into the expected unit of log-Hertz.', category=UserWarning)
    if np.any(inject_index_x < 2):
        raise Exception(error_str+'Injection index in at least one simulation is < 2.')
    if np.any(axis_ratio_x < 1):
        raise Exception(error_str+'Axis ratio in at least one simulation is < 1.')

    # return object and add to array if valid type is provided
    if specz == None:
        calib = False
    object = RAiSERed_obj(source=source, freq=frequency_x, flux=flux_density_x, size=angular_size_x, axis=axis_ratio_x, inj=inject_index_x, brkfreq=break_freq_x, specz=specz, calib=calib, mean=None, stdev=None, x=None, y=None, raw=None, series=series, nredshifts=nredshifts, nsimulations=nsimulations)
    if isinstance(raisered_list, RAiSERed_list):
        raisered_list.append(object)
    else:
        warnings.warn(error_str+'Source not appended to data structure of incorrect type; expect RAiSERed_list.')
        return object
                
# define function to check type of input variables
def __variable_distribution(variable_name, variable, nsimulations):

    if isinstance(variable, (list, np.ndarray)):
        if len(variable) == 3:
            if isinstance(variable[0], (int, float)) and isinstance(variable[1], (int, float)) and isinstance(variable[2], (int, float)):
                variable_x = __random_skewnormal(variable[0], variable[1], variable[2], nsimulations)
            else:
                raise Exception(variable_name+' array elements must be of type float or integer.')
        elif len(variable) == 2:
            if isinstance(variable[0], (int, float)) and isinstance(variable[1], (int, float)):
                variable_x = __random_skewnormal(variable[0], variable[1], 0, nsimulations)
            else:
                raise Exception(variable_name+' array elements must be of type float or integer.')
        elif len(variable) == 1:
            if isinstance(variable[0], (int, float)):
                variable_x = float(variable[0])*np.ones(nsimulations)
            else:
                raise Exception(variable_name+' must be of type float or integer.')
        else:
            raise Exception(variable_name+' must include mean, standard deviation and skewness.')
    elif isinstance(variable, (int, float)):
        variable_x = float(variable)*np.ones(nsimulations)
    else:
        raise Exception(variable_name+' must be of type float or integer.')
        
    return variable_x
    
# define function to define random skew normal distribution
def __random_skewnormal(mean, stddev, skew, n):
   
    if skew == 0:
        # use efficient algorithm for random normal distribution
        x = np.random.normal(mean, stddev, n)
      
        # include only values within 2 sigma
        if isinstance(mean, np.ndarray):
            x[x < mean - 2*stddev] = mean[x < mean - 2*stddev] - 2*stddev
            x[x > mean + 2*stddev] = mean[x > mean + 2*stddev] + 2*stddev
        else:
            x[x < mean - 2*stddev] = mean - 2*stddev
            x[x > mean + 2*stddev] = mean + 2*stddev
    elif skew <= -100:
        # use efficient algorithm for upper limit distribution
        x = np.random.uniform(max(0, mean - 2*stddev), mean, n)
    elif skew >= 100:
        x = np.random.uniform(mean, mean + 2*stddev, n)
    else:
        # randomly sample points in cumulative distribution function
        cdf = np.random.uniform(0.022750132, 0.977249868, n) # include only values within 2 sigma

        # calculate inverse of cumulative distribution function
        x = skewnorm.ppf(cdf, skew)
        # scale inverse function by standard deviation and mean
        x = x*stddev + mean
   
    # return inverse function
    return x


#### RAiSERed function and sub-functions ####
# define parallelised RAiSERed function
def RAiSERed(raisered_list, b1=None, b2=None, b3=None, b4=None):

    # set warnings to run as expected
    warnings.filterwarnings('always')
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    # determine number of sources with spectroscopic data
    calib = []
    nelements = 0
    if isinstance(raisered_list, (list, RAiSERed_list)):
        for element in raisered_list:
            if isinstance(element, RAiSERed_obj):
                calib.append(element.calib)
                nelements = nelements + 1
            else:
                raise Exception('Function input must a list of RAiSERed objects.')
    else:
        raise Exception('Function input must a list of RAiSERed objects.')
    
    # apply adaptive and parallel code to find redshift distribution
    if b1 == None or b2 == None or b3 == None or b4 == None or (not isinstance(b1, (int, float, list, np.ndarray))) or (not isinstance(b2, (int, float, list, np.ndarray))) or (not isinstance(b3, (int, float, list, np.ndarray))) or (not isinstance(b4, (int, float, list, np.ndarray))):
        if sum(np.asarray(calib) == True) < 5:
            warnings.warn('RAiSERed is using default calibration; include at least 5 sources with spectroscopic redshifts to calibrate model.')
            for i in range(0, nelements):
                b1_x, b2_x, b3_x, b4_x = np.zeros(10*raisered_list[i].nsimulations), np.zeros(10*raisered_list[i].nsimulations), np.zeros(10*raisered_list[i].nsimulations), np.zeros(10*raisered_list[i].nsimulations)
                raisered_list[i].mean, raisered_list[i].stdev, raisered_list[i].x, raisered_list[i].y, raisered_list[i].raw = __adaptive_distribution(raisered_list[i].freq, raisered_list[i].flux, raisered_list[i].size, raisered_list[i].brkfreq, raisered_list[i].inj, raisered_list[i].axis, b1_x, b2_x, b3_x, b4_x, nredshifts = raisered_list[i].nredshifts, nsimulations = raisered_list[i].nsimulations)
            return None
        else:
            warnings.warn('RAiSERed is calibrating model using sources with spectroscopic redshifts.')
            # calculate the best fitting normalisation constants b1, b2, b3 and b4
            result = minimizeCompass(__residuals, x0=np.array([0, 0, 0, 0]), args=[raisered_list, nelements], bounds=[[-1, 1], [-1, 1], [-1, 1], [-1, 1]], deltatol=0.01, paired=False, errorcontrol=False)
                            # bounds restricted to a factor of 10 to prevent instability
            mean = result.x
            
            # find fits for non-calibrator sources
            for i in range(0, nelements):
                b1_x = __variable_distribution('b1', mean[0], 10*raisered_list[i].nsimulations)
                b2_x = __variable_distribution('b2', mean[1], 10*raisered_list[i].nsimulations)
                b3_x = __variable_distribution('b3', mean[2], 10*raisered_list[i].nsimulations)
                b4_x = __variable_distribution('b4', mean[3], 10*raisered_list[i].nsimulations)
                raisered_list[i].mean, raisered_list[i].stdev, raisered_list[i].x, raisered_list[i].y, raisered_list[i].raw = __adaptive_distribution(raisered_list[i].freq, raisered_list[i].flux, raisered_list[i].size, raisered_list[i].brkfreq, raisered_list[i].inj, raisered_list[i].axis, b1_x, b2_x, b3_x, b4_x, nredshifts = raisered_list[i].nredshifts, nsimulations = raisered_list[i].nsimulations)
            return mean[0], mean[1], mean[2], mean[3]
    else:
        warnings.warn('RAiSERed is using user specified calibration; include at least 5 sources with spectroscopic redshifts to calibrate model.')
        for i in range(0, nelements):
            b1_x = __variable_distribution('b1', b1, 10*raisered_list[i].nsimulations)
            b2_x = __variable_distribution('b2', b2, 10*raisered_list[i].nsimulations)
            b3_x = __variable_distribution('b3', b3, 10*raisered_list[i].nsimulations)
            b4_x = __variable_distribution('b4', b4, 10*raisered_list[i].nsimulations)
            raisered_list[i].mean, raisered_list[i].stdev, raisered_list[i].x, raisered_list[i].y, raisered_list[i].raw = __adaptive_distribution(raisered_list[i].freq, raisered_list[i].flux, raisered_list[i].size, raisered_list[i].brkfreq, raisered_list[i].inj, raisered_list[i].axis, b1_x, b2_x, b3_x, b4_x, nredshifts = raisered_list[i].nredshifts, nsimulations = raisered_list[i].nsimulations)
        return None

def __residuals(params, raisered_list, nelements):

    b1, b2, b3, b4 = params[0], params[1], params[2], params[3]
    mean, specz = [], []
    distance = False # minimise log(dM) or log(1+z)
    if distance == True:
        cosmo = FlatLambdaCDM(H0=__H0, Om0=__Om0)
    
    for i in range(0, nelements):
        if raisered_list[i].calib == True:
            b1_x, b2_x, b3_x, b4_x = b1*np.ones(10*raisered_list[i].nsimulations), b2*np.ones(10*raisered_list[i].nsimulations), b3*np.ones(10*raisered_list[i].nsimulations), b4*np.ones(10*raisered_list[i].nsimulations)
            raisered_list[i].mean, raisered_list[i].stdev, raisered_list[i].x, raisered_list[i].y, raisered_list[i].raw = __adaptive_distribution(raisered_list[i].freq, raisered_list[i].flux, raisered_list[i].size, raisered_list[i].brkfreq, raisered_list[i].inj, raisered_list[i].axis, b1_x, b2_x, b3_x, b4_x, nredshifts = raisered_list[i].nredshifts, nsimulations = raisered_list[i].nsimulations)
            if distance == True:
                mean.append(math.log10((cosmo.luminosity_distance(raisered_list[i].mean).to(u.Mpc)).value/(1 + raisered_list[i].mean)))
                specz.append(math.log10((cosmo.luminosity_distance(raisered_list[i].specz).to(u.Mpc)).value/(1 + raisered_list[i].specz)))
            else:
                mean.append(math.log10(1 + raisered_list[i].mean))
                specz.append(math.log10(1 + raisered_list[i].specz))

    # return the residual vector
    print ('Current estimate for calibration is b1 = '+'{:.2f}'.format(b1)+', b2 = '+'{:.2f}'.format(b2)+', b3 = '+'{:.2f}'.format(b3)+', b4 = '+'{:.2f}'.format(b4)+'.')
    return ((np.asarray(mean) - np.asarray(specz))**2).sum()

# define adaptive and parallel code to find redshift distribution
def __adaptive_distribution(frequency_x, flux_density_x, angular_size_x, break_freq_x, inject_index_x, axis_ratio_x, b1_x, b2_x, b3_x, b4_x, nredshifts = 50, nsimulations = 10000):

    # initial low-precision run (in log-space) with coarse redshift sampling and n=5000
    ntest = nsimulations
    min_redshift, max_redshift = 0.001, 10
    redshift_r = 10**np.linspace(math.log10(1 + min_redshift), math.log10(1 + max_redshift), nredshifts) - 1

    #if __name__ == "__main__":
    pool = Pool(cpu_count())
    probability_r = pool.map(partial(__probability_distribution, frequency_x = frequency_x[0:ntest], flux_density_x = flux_density_x[0:ntest], angular_size_x = angular_size_x[0:ntest], break_freq_x = break_freq_x[0:ntest], inject_index_x = inject_index_x[0:ntest], axis_ratio_x = axis_ratio_x[0:ntest], b1_x = b1_x[0:ntest], b2_x = b2_x[0:ntest], b3_x = b3_x[0:ntest], b4_x = b4_x[0:ntest], nsimulations = ntest), (z for z in redshift_r))
    pool.close()  # 'TERM'
    pool.join()   # 'KILL'

    probability_r = np.asarray(probability_r)
    absolute_probability = -math.log10(np.max(probability_r))/5 # factor to increase simulation count
    # apply Bayesian prior to probability ditribution
    prior_r = 1./(1 + redshift_r)**4
    probability_r = probability_r*prior_r
    probability_r = probability_r/np.max(probability_r)

    # final high-resolution run (in linear space) with fine redshift sampling and n>5000
    ntest = int(nsimulations*min(10, max(1, absolute_probability)))
    idx = np.asarray(range(0, len(probability_r)))[probability_r > 1e-10] # integer index array
    redshift_r = np.linspace(max(redshift_r[0], 0.75*redshift_r[max(0, np.min(idx) - 1)]), min(redshift_r[-1], 1.25*redshift_r[min(len(redshift_r) - 1, np.max(idx) + 1)]), nredshifts)

    # if __name__ == "__main__":
    pool = Pool(cpu_count())
    probability_r = pool.map(partial(__probability_distribution, frequency_x = frequency_x[0:ntest], flux_density_x = flux_density_x[0:ntest], angular_size_x = angular_size_x[0:ntest], break_freq_x = break_freq_x[0:ntest], inject_index_x = inject_index_x[0:ntest], axis_ratio_x = axis_ratio_x[0:ntest], b1_x = b1_x[0:ntest], b2_x = b2_x[0:ntest], b3_x = b3_x[0:ntest], b4_x = b4_x[0:ntest], nsimulations = ntest), (z for z in redshift_r))
    pool.close()  # 'TERM'
    pool.join()   # 'KILL'
    
    probability_r = np.asarray(probability_r)
    # apply Bayesian prior to probability ditribution
    prior_r = 1./(1 + redshift_r)**4
    probability_r = probability_r*prior_r
    raw_probability_r = probability_r/np.max(probability_r)

    # remove noise from probability distribution
    probability_r = __fourier_distribution(redshift_r, raw_probability_r)

    # find mean and standard deviation of probability distribution
    mean, stdev = __redshift_statistics(redshift_r, probability_r)
         
    return mean, stdev, redshift_r, probability_r/np.max(probability_r), raw_probability_r/10**np.mean(np.log10(raw_probability_r[np.logical_and(probability_r > 1e-10, raw_probability_r > 1e-10)]))*10**np.mean(np.log10(probability_r[np.logical_and(probability_r > 1e-10, raw_probability_r > 1e-10)]))/np.max(probability_r)
                    
# define function to calculate probability distribution
def __probability_distribution(redshift, frequency_x, flux_density_x, angular_size_x, break_freq_x, inject_index_x, axis_ratio_x, b1_x, b2_x, b3_x, b4_x, nsimulations = 10000):
        
    # calculate transverse comoving distance at this redshift
    distance_y = np.zeros((3, 3))
    parity_y = np.zeros((3, 3))
    for j in range(0, 3):
        if j == 0:
            hubble = __H0 - __dH0
        elif j == 1:
            hubble = __H0
        else:
            hubble = __H0 + __dH0
        for k in range(0, 3):
            if j == 0:
                matter = __Om0 - __dOm0
            elif j == 1:
                matter = __Om0
            else:
                matter = __Om0 + __dOm0
            cosmo = FlatLambdaCDM(H0=hubble, Om0=matter)
            dl = (cosmo.luminosity_distance(redshift).to(u.meter)).value
            distance_y[j, k] = np.log10(dl/(1 + redshift)/__Mpc)
            parity_y[j, k] = (j + k + 1)%2
    # calculate uncertainty in transverse comoving distance
    sigma_y = np.sqrt(np.sum((distance_y - distance_y[1, 1])**2/2**parity_y)/4)
                # approximation of error from in H0 and Om0

    # simulate random variations in the equipartition factor
    equi_factor_x = __equipartition(redshift, nsimulations)
    # simulate random variations of the host cluster environment
    radius_x, beta_index_x, density_x = __environment_distribution(redshift, angular_size_x, nsimulations)

    # calculate distance to source using standard ruler
    distance_x = __distance_distribution(redshift, frequency_x, flux_density_x, angular_size_x, break_freq_x, inject_index_x, axis_ratio_x, equi_factor_x, radius_x, beta_index_x, density_x, b1_x, b2_x, b3_x, b4_x)
    idx = np.logical_and(np.isnan(distance_x) == False, np.isinf(distance_x) == False)
    distance_x = np.log10(distance_x[idx])

    # calculate chi-squared statistic and probability for this redshift
    chi_squared_x = (distance_x - distance_y[1, 1])**2/sigma_y**2
    probability_x = np.exp(-chi_squared_x/2) # approximately 2 degrees of freedom
    if len(probability_x) < 10:
        return 0.0
    else:
        idx = np.logical_and(np.quantile(probability_x, 0.022750132) <= probability_x, probability_x <= np.quantile(probability_x, 0.977249868)) # remove extreme values for stability
        probability_x = probability_x[idx]
        
        return np.sum(probability_x)/len(probability_x)
        
# define function to calculate distance to sources
def __distance_distribution(redshift, frequency_x, flux_density_x, angular_size_x, break_freq_x, inject_index_x, axis_ratio_x, equi_factor_x, radius_x, beta_index_x, density_x, b1_x, b2_x, b3_x, b4_x):
            
    # apply restrictions on variables to place in computationally efficient range
    beta_index_x = np.minimum(1.95, beta_index_x)
    inject_index_x = np.maximum(2.05, inject_index_x)

    # convert angular size and flux density to intrinsic properties for this redshift
    cosmo = FlatLambdaCDM(H0=__H0, Om0=__Om0)
    dl = (cosmo.luminosity_distance(redshift).to(u.meter)).value
    size_x = (angular_size_x*__arcseconds)*(dl/(1 + redshift)**2)
    luminosity_x = 4*math.pi*(flux_density_x*__Janskys)*(dl**2)

    # calculate magnetic fields for the source at this redshift for the given environment
    B_val = __B(frequency_x, luminosity_x, size_x, axis_ratio_x, inject_index_x, __adiabatic_loss, __gamma_index, b1_x)
    Bic_val = __Bic(redshift)
    Q_val = __Q(B_val, equi_factor_x, size_x, axis_ratio_x, beta_index_x, density_x, radius_x, b2_x, b3_x)
    velocity_val = (size_x/__age(Q_val, size_x, axis_ratio_x, beta_index_x, density_x, radius_x, b3_x))

    # calculate the exponents and constants
    sigma_val = __sigma(B_val, Bic_val)
    kappa_val = __kappa(B_val, Bic_val, sigma_val)
    x_val = __x(inject_index_x, sigma_val)
    y_val = __y(inject_index_x, beta_index_x, sigma_val)

    f1_val = __f1(inject_index_x, __adiabatic_loss)
    f3_val = __f3(beta_index_x, axis_ratio_x, kappa_val, sigma_val, b3_x)

    # calculate the transverse comoving distance
    distance_x = np.zeros(len(y_val))
    # set values near y = 2, with negative distances, or invalid jet powers to nan
    idx = np.logical_and(angular_size_x > 0, np.logical_and(np.abs(y_val - 2) >= 0.05, np.logical_and(velocity_val < __c, np.logical_and(1e35 <= Q_val, Q_val <= 1e45))))
    distance_x[np.logical_not(idx)] = np.nan
    # set other values to distance
    distance_x[idx] = (axis_ratio_x[idx]**2*(10**b1_x[idx]*__gamma_index)**(2 - inject_index_x[idx])*(frequency_x[idx])**((inject_index_x[idx] - 1)/2)*(flux_density_x[idx]*__Janskys)/f1_val[idx])**(-1./(2 - y_val[idx]))*(10**b2_x[idx]*f3_val[idx]*(radius_x[idx]*__kpc)**beta_index_x[idx]*density_x[idx]*(1 + redshift)**(b4_x[idx])*(equi_factor_x[idx]/(equi_factor_x[idx] + 1))*(10**break_freq_x[idx]))**(x_val[idx]/(2 - y_val[idx]))*(angular_size_x[idx]*__arcseconds)**(y_val[idx]/(2 - y_val[idx]))*(1 + redshift)**((x_val[idx] - y_val[idx] - 2)/(2 - y_val[idx]))/__Mpc

    return distance_x
    
# define physical constants as private variables
__c, __mu0 = 299792458., 1.25664e-06
__echarge, __me, __sigmaT = 1.60218e-19, 9.10938e-31, 6.65246e-29
__arcseconds, __Janskys = 4.84814e-06, 1e-26
__km, __kpc, __Mpc = 1000., 3.08568e+19, 3.08568e+22
__gravconst, __solar, __yrs = 6.67408e-11, 1.98847e+30, 365.25*24*3600
__GammaC, __GammaX = 4./3, 5./3
__H0, __dH0, __Om0, __dOm0 = 67.74, 0.46, 0.3089, 0.0062

__adiabatic_loss = 0.4
__gamma_index = 288
__upsilon = (243*math.pi*__me**5*__c**2/(4*__mu0**2*__echarge**7))**(1/2.)

# define basic function used in analytic distance calculation
def __f1(inject_index_x, adiabatic_loss):
    return __sigmaT*(inject_index_x - 2)/(9*__me*__c)*(__echarge**2*__mu0/(2*math.pi**2*__me**2))**((inject_index_x - 3)/4)*adiabatic_loss

def __f2(beta_index_x, axis_ratio_x, b3_x):
    chi_val = __chi(beta_index_x, axis_ratio_x, b3_x)
    return 18*chi_val/((__GammaX + 1)*(5 - beta_index_x)**2*(__GammaC - 1))

def __f3(beta_index_x, axis_ratio_x, kappa_val, sigma_val, b3_x):
    return __f2(beta_index_x, axis_ratio_x, b3_x)*(2*__mu0)**(-sigma_val)*(kappa_val**4/__upsilon**2)

def __chi(beta_index_x, axis_ratio_x, b3_x):
    return 1./((2.14 - 0.52*beta_index_x)*(axis_ratio_x/2)**(2.04 - 0.25*(beta_index_x + b3_x)))

def __B(frequency_x, luminosity_x, size_x, axis_ratio_x, inject_index_x, adiabatic_loss, gamma_index, b1_x):
    f1_val = __f1(inject_index_x, adiabatic_loss)
    rhs = 4*math.pi*f1_val*(frequency_x)**(-(inject_index_x - 1)/2)/(axis_ratio_x**2*(10**b1_x*gamma_index)**(2 - inject_index_x))*(1./(2*__mu0))**((inject_index_x + 5)/4)*size_x**3
    return (luminosity_x/rhs)**(2/(inject_index_x + 5))

def __Bic(redshift):
    return 0.318e-9*(1 + redshift)**2

def __c1(beta_index_x, axis_ratio_x, b3_x):
    chi_val = __chi(beta_index_x, axis_ratio_x, b3_x)
    return (chi_val**(-1./__GammaC)*(__GammaC*(__GammaC - 1)*(__GammaX + 1)*(5 - beta_index_x)**3)/(18*(2*math.pi/(3*axis_ratio_x**2))*(9*__GammaC - 4 - beta_index_x)))**(1./(5 - beta_index_x))

def __Q(B_val, equi_factor_x, size_x, axis_ratio_x, beta_index_x, density_x, radius_x, b2_x, b3_x):
    c1_val = __c1(beta_index_x, axis_ratio_x, b3_x)
    f2_val = __f2(beta_index_x, axis_ratio_x, b3_x)
    return ((B_val**2*(equi_factor_x + 1)*size_x**((4 + beta_index_x)/3))/(2*__mu0*(10**b2_x)*f2_val*equi_factor_x*c1_val**(2*(5 - beta_index_x)/3)*(density_x*(radius_x*__kpc)**beta_index_x)**(1./3)))**(3./2)

def __age(Q_val, size_x, axis_ratio_x, beta_index_x, density_x, radius_x, b3_x):
    c1_val = __c1(beta_index_x, axis_ratio_x, b3_x)
    return (size_x/(c1_val*(radius_x*__kpc)))**((5 - beta_index_x)/3.)*(((radius_x*__kpc)**5*density_x)/Q_val)**(1./3)

def __sigma(B_val, Bic_val):
    return (-3*B_val**2 + Bic_val**2)/(2*(B_val**2 + Bic_val**2))

def __kappa(B_val, Bic_val, sigma_val):
    return np.sqrt(B_val**(sigma_val - 0.5)*(B_val**2 + Bic_val**2))

def __x(inject_index_x, sigma_val):
    return (inject_index_x + 5)/(4*(sigma_val + 1))

def __y(inject_index_x, beta_index_x, sigma_val):
    return (22 + 12*sigma_val + 2*inject_index_x - 5*beta_index_x - inject_index_x*beta_index_x)/(4*(sigma_val + 1))

# define constants to simulate environment and equipartition factors
__alphaAvg, __alphaStdev = 1.64, 0.30
__betaPrimeAvg, __betaPrimeStdev = 0.56, 0.10
__gammaPrimeAvg, __gammaPrimeStdev = 3, 0
__epsilonAvg, __epsilonStdev = 3.23, 0
__rCoreAvg, __rCoreStdev = 0.087, 0.028
__rSlopeAvg, __rSlopeStdev = 0.73, 0

__halogasfracCONST1z0, __halogasfracCONST1z1 = (-0.881768418), (-0.02832004)
__halogasfracCONST2z0, __halogasfracCONST2z1 = (-0.921393448), 0.00064515
__halogasfracSLOPE = 0.053302276
__dhalogasfracz0, __dhalogasfracz1 = 0.05172769, (-0.00177947)
__SAGEdensitycorr = (-0.1)

__equiFactorAvg, __equiFactorStdev = 1.73, 0.10

# define functions to simulate environment and equipartition factors
def __random_halomass(redshift, nsimulations):

    # define range of possible halo masses
    min_mass, max_mass, delta_mass = 11.5, 15.5, 0.01
    mass_bins = 10**np.arange(min_mass, max_mass + 0.1*delta_mass, delta_mass)

    # Giraldi & Giuricin (2000) halo mass distribution and Pope et al. (2012) AGN duty cycle
    mass_counts = (mass_bins/(3.1e14/(__H0/100.)/(1 + redshift)**3))**(-1.55)*np.exp(-mass_bins/(3.1e14/(__H0/100.)/(1 + redshift)**3))
    AGN_counts = mass_counts*mass_bins**1.5

    # calculate cumulative probability distribution
    cumulative = np.cumsum(AGN_counts)
    cumulative = cumulative/cumulative[-1]

    # randomly sample masses from the probability distribution
    random_x = np.random.uniform(0, 1, nsimulations)
    difference_x = (random_x.reshape(1, -1) - cumulative.reshape(-1, 1))
    idx = np.abs(difference_x).argmin(axis=0)
    halo_mass_x = mass_bins[idx]
                
    return halo_mass_x

def __density_distribution(virial_fraction_r, virial_radius_x, alpha, betaPrime, gammaPrime, epsilon, rCore, rSlope, dimensions = 1):
    
    if dimensions == 2:
        radius_rx = np.outer(virial_fraction_r, virial_radius_x)
    else:
        radius_rx = virial_fraction_r*virial_radius_x
    
    return np.sqrt((radius_rx/(rCore*virial_radius_x))**(-alpha)/((1 + radius_rx**2/(rCore*virial_radius_x)**2)**(3*betaPrime - alpha/2)*(1 + radius_rx**gammaPrime/(rSlope*virial_radius_x)**gammaPrime)**(epsilon/gammaPrime)))

def __environment_distribution(redshift, angular_size_x, nsimulations):

    cosmo = FlatLambdaCDM(H0=__H0, Om0=__Om0)
    dl = (cosmo.luminosity_distance(redshift).to(u.meter)).value
    linear_size_x = (angular_size_x*__arcseconds)*(dl/(1 + redshift)**2)
    
    # call function describing host halo masses prior probability distribution
    halo_mass_x = __random_halomass(redshift, nsimulations)

    # calculate corresponding virial radius
    virial_radius_x = (halo_mass_x*__solar/(100/__gravconst*(__H0*np.sqrt(__Om0*(1 + redshift)**3 + (1 - __Om0))*__km/__Mpc)**2))**(1.0/3)
    
    # calculate corresponding gas mass
    halogasfrac = np.maximum(__halogasfracCONST1z0 + __halogasfracCONST1z1*redshift, __halogasfracCONST2z0 + __halogasfracCONST2z1*redshift) + __halogasfracSLOPE*(np.log10(halo_mass_x) - 14)
    dhalogasfrac = __dhalogasfracz0 + __dhalogasfracz1*redshift
    gas_mass_x = halo_mass_x*10**(__random_skewnormal(halogasfrac, dhalogasfrac, 0, nsimulations))
    
    # randomly generate density profile
    alpha = __random_skewnormal(__alphaAvg, __alphaStdev, 0, nsimulations)
    betaPrime = __random_skewnormal(__betaPrimeAvg, __betaPrimeStdev, 0, nsimulations)
    gammaPrime = __random_skewnormal(__gammaPrimeAvg, __gammaPrimeStdev, 0, nsimulations)
    epsilon = __random_skewnormal(__epsilonAvg, __epsilonStdev, 0, nsimulations)
    rCore = __random_skewnormal(__rCoreAvg, __rCoreStdev, 0, nsimulations)
    rSlope = __random_skewnormal(__rSlopeAvg, __rSlopeStdev, 0, nsimulations)
    
    # calculate scaling of density profile
    delta_fraction = 0.01
    virial_fraction_r = 10**np.arange(-3, 1e-9, delta_fraction)
    density_rx = __density_distribution(virial_fraction_r, virial_radius_x, alpha, betaPrime, gammaPrime, epsilon, rCore, rSlope, dimensions=2)
    unscaled_mass = 4*math.pi*np.sum(density_rx*np.outer(virial_fraction_r, virial_radius_x)**3, axis=0)*np.log(10)*delta_fraction
    core_density_x = gas_mass_x*__solar/unscaled_mass
    
    # calculate density and exponent (beta) at angular size of source
    factor = 10 # factor in radius over which to calculate beta index
    density_x = core_density_x*__density_distribution(linear_size_x/virial_radius_x, virial_radius_x, alpha, betaPrime, gammaPrime, epsilon, rCore, rSlope)
    density_0 = core_density_x*__density_distribution(linear_size_x/virial_radius_x/factor, virial_radius_x, alpha, betaPrime, gammaPrime, epsilon, rCore, rSlope)
    beta_index_x = -np.log10(density_x/density_0)/np.log10(factor)

    return linear_size_x/__kpc, beta_index_x, density_x

def __equipartition(redshift, nsimulations):
    
    return 10**__random_skewnormal(-__equiFactorAvg, __equiFactorStdev, 0, nsimulations)

# define sub-functions used in RAiSERed function
def __fourier_distribution(redshift, probability):

    # take Fourier transform of probabilities
    fourier = np.zeros(len(probability))
    w = scipy.fftpack.rfft(np.log10(probability[probability > 0]))

    # convolve with exponential function to reduce amplitude of high-frequency terms
    f = np.linspace(0, 1, len(w))
    w2 = w*np.exp(-4.5*f)

    # take inverse-Fourier transform
    y2 = scipy.fftpack.irfft(w2)

    fourier[probability > 0] = 10**(y2)
    return fourier
    
def __redshift_statistics(redshifts, probabilities):

    # calculate peak value and uncertainties
    mean = np.sum(redshifts*probabilities)/np.sum(probabilities)
    stdev = np.sqrt(np.sum(((redshifts - mean)*probabilities)**2)/np.sum(probabilities**2))

    return mean, stdev


#### RAiSERed_plot function and sub-functions ####
# define functions used to plot probability distributions
def RAiSERed_plot(raisered_list):
    
    warnings.filterwarnings('ignore', category=UserWarning)

    # unpack RAiSERed object
    if isinstance(raisered_list, RAiSERed_obj):
        return __redshift_plotter(raisered_list.source, raisered_list.x, raisered_list.y, raisered_list.raw).figure
    elif isinstance(raisered_list, (list, RAiSERed_list)):
        plt_list = []
        for element in raisered_list:
            if isinstance(element, RAiSERed_obj):
                plt_list.append(__redshift_plotter(element.source, element.x, element.y, element.raw).figure)
            else:
                raise Exception('Function input must be a RAiSERed object or a list of RAiSERed objects.')
        return plt_list
    else:
        raise Exception('Function input must be a RAiSERed object or a list of RAiSERed objects.')

def __redshift_plotter(source, redshift, probability, raw_probability):
    
    # set maximum redshift on plot
    if np.max(redshift) < 1:
        max_redshift = math.ceil(np.max(redshift)*4 + 0.25)/4.
    elif np.max(redshift) < 2.5:
        max_redshift = math.ceil(np.max(redshift)*2 + 0.25)/2.
    elif np.max(redshift) < 5:
        max_redshift = math.ceil(np.max(redshift) + 0.25)*1.
    else:
        max_redshift = math.ceil(np.max(redshift)/2 + 0.25)*2.
    
    # define size and axes of plot
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(7.5, 5.5))

    rc('text', usetex=True)
    rc('font', size=14)
    rc('legend', fontsize=14)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    # set axes labels and scale
    axes.set_xlabel(r'Redshift')
    axes.set_ylabel(r'Probability')
    
    axes.set_yscale('log')
    axes.set_xlim((0, max_redshift))
    axes.set_ylim((1e-6, 10))
    axes.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    
    # plot lines on plot
    axes.fill_between(redshift, probability, color = 'whitesmoke')

    axes.plot([0, max_redshift], [math.exp(-4./2), math.exp(-4./2)], color = 'grey', linestyle = 'dashed', linewidth = 1)
    axes.plot([0, max_redshift], [math.exp(-9./2), math.exp(-9./2)], color = 'grey', linestyle = 'dashed', linewidth = 1)
    axes.plot([0, max_redshift], [math.exp(-25./2), math.exp(-25./2)], color = 'grey', linestyle = 'dashed', linewidth = 1)
    
    axes.plot(redshift, raw_probability, color = 'crimson', linewidth = 0.75)
    axes.plot(redshift, probability, color = 'black', linewidth = 1.25, label=source)
    
    # add source name to plot
    if not source == None:
        axes.legend(handlelength=0, handletextpad=0, loc='upper right', frameon=False)
    
    # return plot handle to enable further modification
    return axes


#### RAiSERed_zzplot function and sub-functions ####
# define functions to create redshift-redshift violin plots
def RAiSERed_zzplot(raisered_list, legend_text=None, plot_type='default'):

    warnings.filterwarnings('ignore', category=UserWarning)

    # unpack RAiSERed object
    if isinstance(raisered_list, RAiSERed_obj):
        raise Exception('List of RAiSERed objects must contain at least two spectroscopic redshifts.')
    elif isinstance(raisered_list, (list, RAiSERed_list)):
        name_list = []
        series_list = []
        z_list = []
        specz_list = []
        for element in raisered_list:
            if isinstance(element, RAiSERed_obj):
                if isinstance(element.specz, (int, float)):
                    x = element.x
                    dx = (x[1] - x[0])/2
                    y = element.y
                    n = np.sum(y)/10000. # 10000 probabilities
                    for i in range(0, len(x)):
                        if i == 0:
                            z_dist = np.random.uniform(x[i] - dx, x[i] + dx, int(y[i]/n))
                        else:
                            z_dist = np.append(z_dist, np.random.uniform(x[i] - dx, x[i] + dx, int(y[i]/n)))
                    name_list.append(element.source)
                    series_list.append(element.series)
                    z_list.append(1 + z_dist[np.logical_and(np.quantile(z_dist, 0.00135) < z_dist, z_dist < np.quantile(z_dist, 0.99865))])
                    specz_list.append(1 + element.specz)
            else:
                raise Exception('Function input must be a RAiSERed object or a list of RAiSERed objects.')
        if len(specz_list) >= 2:
            return __redshift_zzplotter(name_list, series_list, z_list, specz_list, legend_text=legend_text, \
                         plot_type=plot_type).figure
        else:
            raise Exception('List of RAiSERed objects must contain at least two spectroscopic redshifts.')
    else:
        raise Exception('Function input must be a RAiSERed object or a list of RAiSERed objects.')
    
def __redshift_zzplotter(name_list, series_list, redshift_list, specz_list, legend_text=None, plot_type='default'):

    # set maximum redshift on plot
    if np.max(specz_list) < 1:
        max_redshift = math.ceil((np.max(specz_list) - 1)*4 + 0.25)/4. + 1
    elif np.max(specz_list) < 2.5:
        max_redshift = math.ceil((np.max(specz_list) - 1)*2 + 0.25)/2. + 1
    elif np.max(specz_list) < 5:
        max_redshift = math.ceil((np.max(specz_list) - 1) + 0.25)*1. + 1
    else:
        max_redshift = math.ceil((np.max(specz_list) - 1)/2 + 0.25)*2. + 1

    # define size and axes of plot
    fig, axes = plt.subplots(1, 1, sharex=True, figsize=(6.5, 6.25))

    rc('text', usetex=True)
    rc('font', size=14)
    rc('legend', fontsize=14)
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

    # set axes labels and scale
    axes.set_xlabel(r'Spectroscopic redshift')
    axes.set_ylabel(r'Radio continuum redshift')

    axes.set_xlim((1, max_redshift))
    axes.set_ylim((1, max_redshift))
    axes.set_aspect('equal')

    # plot in log scale
    axes.set_xscale('log')
    axes.set_yscale('log')

    minors = np.append(np.arange(1, 1.5, 0.05), np.append(np.arange(1.5, 3, 0.1), np.arange(3, 11, 0.5)))
    majors = np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    axes.xaxis.set_minor_locator(ticker.FixedLocator(minors))
    axes.xaxis.set_major_locator(ticker.FixedLocator(majors))
    axes.yaxis.set_minor_locator(ticker.FixedLocator(minors))
    axes.yaxis.set_major_locator(ticker.FixedLocator(majors))

    axes.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, majors: '{0:g}'.format(x-1)))
    axes.xaxis.set_minor_formatter(ticker.NullFormatter())
    axes.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x-1)))
    axes.yaxis.set_minor_formatter(ticker.NullFormatter())

    # plot lines on plot
    axes.plot([0, max_redshift], [0, max_redshift], color = 'grey', linestyle = 'dashed', linewidth = 1, \
                zorder=0, label='_nolegend_')

    # split bars in half for sources with east/west or north/south lobes
    redshift1_list, redshift2_list, redshift3_list = [], [], []
    specz1_list, specz2_list, specz3_list = [], [], []
    if plot_type == 'lobes':
        for i in range(0, len(specz_list)):
            if name_list[i][-1] == 'E' or name_list[i][-1] == 'N':
                redshift2_list.append(redshift_list[i])
                specz2_list.append(specz_list[i])
            elif name_list[i][-1] == 'W' or name_list[i][-1] == 'S':
                redshift1_list.append(redshift_list[i])
                specz1_list.append(specz_list[i])
            else:
                redshift3_list.append(redshift_list[i])
                specz3_list.append(specz_list[i])
    elif plot_type == 'series':
        for i in range(0, len(specz_list)):
            if series_list[i] == 1:
                redshift1_list.append(redshift_list[i])
                specz1_list.append(specz_list[i])
            elif series_list[i] == 2:
                redshift2_list.append(redshift_list[i])
                specz2_list.append(specz_list[i])
            else:
                redshift3_list.append(redshift_list[i])
                specz3_list.append(specz_list[i])
    else:
        redshift1_list = redshift_list
        specz1_list = specz_list

    v1 = axes.violinplot(redshift1_list, specz1_list, widths=0.1*np.asarray(specz1_list), showextrema=False)
    if len(specz2_list) > 0:
        v2 = axes.violinplot(redshift2_list, specz2_list, widths=0.1*np.asarray(specz2_list), showextrema=False)
    if len(specz3_list) > 0:
        v3 = axes.violinplot(redshift3_list, specz3_list, widths=0.1*np.asarray(specz3_list), showextrema=False)
    for b in v1['bodies']:
        b.set_facecolor('mediumblue')
        b.set_edgecolor('mediumblue')
        b.set_alpha(0.6)
        if len(specz2_list) > 0 and plot_type == 'lobes':
            m = np.mean(b.get_paths()[0].vertices[:, 0])
            b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
    if len(specz2_list) > 0:
        for b in v2['bodies']:
            b.set_facecolor('darkorchid')
            b.set_edgecolor('darkorchid')
            b.set_alpha(0.6)
            if plot_type == 'lobes':
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
    if len(specz3_list) > 0:
        if plot_type == 'lobes':
            for b in v3['bodies']:
                b.set_facecolor('blue')
                b.set_edgecolor('blue')
                b.set_alpha(0.6)
        else:
            for b in v3['bodies']:
                b.set_facecolor('crimson')
                b.set_edgecolor('crimson')
                b.set_alpha(0.6)
            
    quartile1, medians, quartile3 = [], [], []
    for i in range(0, len(specz_list)):
        quartile1.append(np.quantile(redshift_list[i], 0.16))
        medians.append(np.quantile(redshift_list[i], 0.5))
        quartile3.append(np.quantile(redshift_list[i], 0.84))

    axes.vlines(specz_list, quartile1, quartile3, color='black', alpha=0.8, linestyle='-', lw=2.5, label='_nolegend_')

    if not legend_text == None:
        if isinstance(legend_text, (list, np.ndarray)):
            if plot_type == 'lobes' and len(legend_text) == 2:
                axes.legend([v2['bodies'][0], v1['bodies'][0]], legend_text[0:2], loc='upper left')
            elif plot_type == 'series' and len(legend_text) == 2:
                axes.legend([v1['bodies'][0], v2['bodies'][0]], legend_text[0:2], loc='upper left')
            elif plot_type == 'series' and len(legend_text) == 3:
                axes.legend([v1['bodies'][0], v2['bodies'][0], v3['bodies'][0]], legend_text[0:3], loc='upper left')
            else:
                axes.legend([v1['bodies'][0]], legend_text, loc='upper left')
        elif isinstance(legend_text, str):
            axes.legend([v1['bodies'][0]], legend_text, loc='upper left')

    # return plot handle to enable further modification
    return axes
