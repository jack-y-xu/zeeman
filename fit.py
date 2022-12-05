import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.optimize
import scipy.signal
from scipy.stats import norm
import random
import string

def get_intensity_errors(intensities, smoothed, plot: bool = True):
    errs = intensities - smoothed
    errs = errs - np.mean(errs)

    if plot:
        plt.hist(errs, bins=40)
        plt.title(np.std(errs))
        plt.show()

    return np.std(errs)


def gaussian_filter(arr: np.array, sigma: float, order: int = 0) -> np.array:
    """
    Adds a Gaussian filter to array
    """
    return scipy.ndimage.gaussian_filter1d(arr, sigma, order=order)

def find_peaks(arr: np.array, **kwargs):
    """
    Args: An array
    Does: Finds the peaks of noisy data using scipy signals
    Does (optional): Plots the 
    Returns: A list of peaks
    """
    return scipy.signal.find_peaks(arr, **kwargs)

def find_peaks_cwt(arr: np.array, **kwargs):
    """
    Args: An array
    Does: Finds the peaks of noisy data using continuous wavelet transform
    Returns: A list of peaks
    """
    return scipy.signal.find_peaks_cwt(arr, **kwargs)

def gaussian_func(x, a, mu, sigma, offset):
    return a*np.exp(-np.power(x-mu, 2)/(2*sigma**2)) + offset


def find_single_peak_bootstrap(arr: np.array, err: float, plot: bool = False, iterations: int = 100, bootstrap: bool = False):
    """
    Function for finding a single peak
    Args: 
        arr - array concerned
    Does:
        Fits a Gaussian curve using scipy. Optionally plots it.
    Returns:
        mu, sigma
    """

    x = np.array(range(len(arr)))
    y = arr
    mu_0 = sum(x*y)/sum(y)
    sigma_0 = np.sqrt(sum(y*(x-mu_0)**2) / sum(y))

    popt_0, pcov_0 = scipy.optimize.curve_fit(gaussian_func, x, y, sigma=np.ones_like(y)*err, p0=[max(y), mu_0, sigma_0, min(y)], absolute_sigma=True, maxfev=10000)

    if bootstrap:
        popts = None

        for i in range(iterations):

            noise = np.random.normal(0, err, arr.shape)
            x = np.array(range(len(arr)))
            y = arr + noise
            mu_0 = sum(x*y)/sum(y)
            sigma_0 = np.sqrt(sum(y*(x-mu_0)**2) / sum(y))

            popt, pcov = scipy.optimize.curve_fit(gaussian_func, x, y, p0=[max(y), mu_0, sigma_0, min(y)], maxfev=10000)

            if popts is None:
                popts = popt
            else:
                popts = np.vstack([popts, popt])
        
        std_popt = np.std(popts, axis=0)
    
    if plot:
        plt.plot(x, arr)
        plt.plot(x, gaussian_func(x, *popt_0))
        plt.show()
    
    params = list(popt_0)

    if bootstrap:
        params.append(std_popt[1])
    else:
        params.append(np.sqrt(pcov_0[1,1]))
    
    return params

def plot_intensity(arr: np.array, maxima_x: np.array = None, maxima_y: np.array = None, save: bool = True, save_path: str = 'scratch', save_name: str = None):
    """
    Args: 
        Array for intensity to be plotted
    Optional Args:
        maxima_x: x coordinates of maxima to be plotted
        maxima_y: y coordinates of maxima to be plotted
        save: whether to save the plot
        save_name: what name to save the plot under

    Does: Plots and saves (optionally)
    Returns: None

    NOTE: This is not meant to be used for graphs for reports. It is mostly meant to document graphs I've made.
    To make graphs for the report, go use a python notebook
    """

    plt.plot(range(len(arr)), arr)

    if maxima_x != None and maxima_y != None:
        plt.scatter(maxima_x, maxima_y, c='r')

    if save:
        if save_name:
            plt.savefig(save_path + '/' + save_name)

        else:
            plt.savefig(save_path + '/' + "bro leave a legit name " + ''.join(random.choices(string.ascii_lowercase, k=5)))
    
    plt.show()

    