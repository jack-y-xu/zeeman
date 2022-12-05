# This file should be able to take the locations of the maximas
# for images of a given spectral line, and with some user input,
# Correspond them. That is, match original lines to their split lines
# Then, it should be able to save the resultant matches into a file


import glob
import re
import pandas as pd
import numpy as np
import configparser
import preprocess

CONFIG_FILE = '/Users/jackxu/Library/Mobile Documents/com~apple~CloudDocs/Document/Code/Zeeman/params.conf'

__DEBUG__ = False
__CONFIGGED__ = False

# Parameters for calculating magnetic field from voltage
__x_const__ = None
__x_scale__ = None
__y_const__ = None
__y_scale__ = None
__err__ = None
__a__ = None
__b__ = None
__c__ = None
__d__ = None

def debug(arg):
    global __DEBUG__
    if __DEBUG__:
        print(arg)

def load_line(rel_path, element, line, suffix):

    split_lines = glob.glob(f"{rel_path}/{element}_{line}_*_{suffix}")
    split_lines = sorted(split_lines, key=lambda x: preprocess.voltage_from_filepath(x))
    base_file = split_lines[0]
    split_lines.remove(base_file)

    df = pd.read_csv(base_file, header=None)
    df.columns = ["0V", "0V_err"]

    voltages = [preprocess.voltage_from_filepath(file) for file in split_lines]

    for file, voltage in zip(split_lines, voltages):

        vdf = pd.read_csv(file, header=None)

        vdf = pd.DataFrame(np.hstack([vdf.iloc[0::2, :].values, vdf.iloc[1::2, :].values]))

        vdf.columns = [f"{voltage}_l1", f"{voltage}_l1_err", f"{voltage}_l2", f"{voltage}_l2_err"]

        df = pd.concat([df, vdf], axis=1)

    return df

def compute_k(df, save_path):
    """
    Input: dataframe and voltages as outputted in load_line
    Output: Computed dk values for each voltage and line
    """

    voltages = df.columns[df.columns.str.contains('_l1_err')].str[:-7].values

    # Compute L, dL
    for i in [1,2]:
        df[f'l{i}_L'] = df['0V'].shift(1-i) - df['0V'].shift(2-i)
        df[f'l{i}_L_err'] = np.sqrt(df['0V_err'].shift(1-i)**2 + df['0V_err'].shift(2-i)**2)

    # compute x
    x_df = df.filter(regex='_l[12]$')
    x_df = abs(x_df.subtract(df['0V'], axis=0)).add_prefix('x_')

    # compute dx
    x_err_df = df.filter(regex='_l[12]_err$')
    x_err_df = np.sqrt((x_err_df**2).add(df['0V_err']**2, axis=0))

    # compute x/L
    xl_df = x_df.copy()

    for i in [1,2]:
        i_cols = x_df.columns[x_df.columns.str.contains(f'_l{i}')]
        xl_df[i_cols] = x_df[i_cols].div(df[f'l{i}_L'], axis=0)
    
    xl_df.columns = xl_df.columns.str[2:] + '_x_l_ratio'

    # compute error in (x/L)

    # compute error ratio of x
    x_cols = x_df.columns.str[2:]
    x_err_cols = x_err_df.columns.str[:-4]
    assert(all(x_cols == x_err_cols))
    dx_x_df = x_err_df.divide(x_df.values, axis=0)
    dx_x_df.columns = dx_x_df.columns.str[:-4] + '_dx_x_ratio'

    # compute error ratio of l
    dl_l_df = pd.DataFrame()
    for i in ['1','2']:
        dl_l_df[f"dl{i}_L"] = df[f"l{i}_L_err"]/df[f"l{i}_L"]

    # compute error ratio for each line
    err_ratio_df = dx_x_df.copy()
    err_ratio_df.columns = err_ratio_df.columns.str[:-10]+'err_ratio'

    for i in ['1','2']:
        i_cols = err_ratio_df.columns[dx_x_df.columns.str.contains(f'_l{i}_')]
        err_ratio_df[i_cols] = np.sqrt((err_ratio_df[i_cols]**2).add(dl_l_df[f'dl{i}_L']**2, axis=0))

    # compute errors for each line
    assert(all(xl_df.columns.str[:-10] == err_ratio_df.columns.str[:-10]))
    err_df = xl_df.mul(err_ratio_df.values, axis=0).add_suffix('_err')

    # stack l1, l2 together

    k_df = pd.concat([xl_df, err_df], axis=1)
    k_arr = np.concatenate([k_df.iloc[:,::2].values, k_df.iloc[:,1::2].values])
    k_df = pd.DataFrame(k_arr, columns = np.concatenate([voltages, voltages+'_err']))

    k_errs = k_df[voltages + '_err']
    ks = k_df[voltages]

    final_k = ks.mul(1/k_errs.values**2, axis=0).sum()/np.nansum(1/k_errs.values**2,axis=0)
    final_k_errs = 1/np.sqrt(np.nansum(1/k_errs.values**2,axis=0))

    k_means = ks.mean()
    k_stds = ks.std()

    results_df = pd.DataFrame(k_means)
    results_df['k_err'] = k_stds
    results_df.reset_index(inplace=True)
    results_df.columns = ['Voltage', 'k', 'k_err']
    results_df = results_df.astype(float)
    results_df['B'], results_df['B_err'] = voltage_to_magfield(results_df['Voltage'])

    results_df.to_csv(save_path)
    return results_df





def __initialize__():
    __CONFIGGED__ = True

    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)

    debug(config.sections())

    global __a__
    global __b__
    global __c__
    global __d__
    global __err__

    __a__ = float(config['cubic_params']['a'])
    __b__ = float(config['cubic_params']['b'])
    __c__ = float(config['cubic_params']['c'])
    __d__ = float(config['cubic_params']['d'])
    __err__ = float(config['cubic_params']['err'])

    global __x_const__ 
    global __x_scale__
    global __y_const__
    global __y_scale__

    __x_const__ = float(config['scaling_params']['x_const'])
    __x_scale__ = eval(config['scaling_params']['x_scale'])
    __y_const__ = float(config['scaling_params']['y_const'])
    __y_scale__ = eval(config['scaling_params']['y_scale'])


def voltage_to_magfield(voltages):
    """
    Converts voltage to magnetic field.
    Returns (magnetic fields, error of each)
    """

    global __err__
    global __CONFIGGED__

    if __CONFIGGED__ == False:
        __initialize__()

    return __scale_back__(__cubic_func__(__normalize__(voltages))), __err__

def __cubic_func__(x):
    global __a__
    global __b__
    global __c__
    global __d__
    return __a__ * np.power(x,3) + __b__ * np.power(x,2) + __c__ * np.power(x,1) + __d__

def __normalize__(x):
    global __x_const__ 
    global __x_scale__
    return x*__x_scale__ + __x_const__

def __scale_back__(y):
    global __y_const__
    global __y_scale__
    return (y+__y_const__) * __y_scale__




