# This file controls all of the analysis of the Zeeman effect

# Parameters
# -e Element, e.g. Hg
# -p Path (e.g. "Jack Zeeman Data/day 2")
# -l Line in nm (e.g. 579)
# -preprocess Rotating images and saving them
# -fit 

import sys, os, argparse, re
import combine, fit, preprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

parser = argparse.ArgumentParser()

parser.add_argument('--preprocess', action='store_true')
parser.add_argument('--fit', action='store_true')
parser.add_argument('--combine', action='store_true')
parser.add_argument('--fit_graph', action='store_true')
parser.add_argument('--debug', action='store_true')

parser.add_argument('-e', type=str, required=True)
parser.add_argument('-l', type=int, required=True)
parser.add_argument('-p', type=str, required=True)

args = parser.parse_args()

def debug(s):
    if args.debug:
        print(s)

if __name__ == '__main__':

    element = args.e
    line = args.l
    path = args.p
    save_path = f"analysis/{element}_{line}"

    os.makedirs(save_path, exist_ok=True)


    if args.preprocess:
        image_paths = preprocess.get_image_filepaths(path, element, line)
        rotated_image_paths = preprocess.rotate_and_crop_images(image_paths, save_path=save_path)
        intensities_file_list = [preprocess.average_vertical_intensity(i_path, save_path=save_path) for i_path in rotated_image_paths]
        intensities_list_path = f"{save_path}/{element}_{line}_intensities_list"
        with open(intensities_list_path, 'wb') as fp:
            pickle.dump(intensities_file_list, fp)
    else:

        intensities_list_path = f"{save_path}/{element}_{line}_intensities_list"
        with open(intensities_list_path, 'rb') as fp:
            intensities_file_list = pickle.load(fp)

    if args.fit:

        intensities_file_list.sort(key=lambda x: preprocess.voltage_from_filepath(x))
        debug(intensities_file_list)

        # Change cutoff

        intensities = np.load(intensities_file_list[0])

        plt.plot(list(range(len(intensities))), intensities)
        plt.show()

        while True:
            cutoff = int(input("Input cutoff value: "))
            plt.plot(list(range(len(intensities))), intensities)
            plt.axvline(x = cutoff)
            plt.show()
            if preprocess.get_bool("Satisfied?"):
                break

        x_range = list(range(cutoff))

        for file in intensities_file_list:

            filename = preprocess.get_filename(file)

            intensities = np.load(file)[:cutoff]

            plt.plot(list(range(len(intensities))), intensities)
            plt.show()

            if preprocess.get_bool("Skip file?"):
                continue
            
            # Change smoothing parameters 
            while True:
                sigma = float(input("Smoothing parameter sigma: "))
                smoothed = fit.gaussian_filter(intensities, sigma)
                plt.plot(x_range, smoothed)
                plt.show()
                if preprocess.get_bool("Satisfied?"):
                    plt.plot(x_range, smoothed)
                    plt.title(f"Smoothing for {filename} with sigma={sigma}\nand cutoff={cutoff}")
                    plt.savefig(f"{save_path}/{filename[:-4]}_smoothed.png")
                    plt.clf()
                    break

            # Find locations of peak using find_peaks, adjust parameters until good

            while True:
                peak_width = int(input("Peak width for scipy peak detection: "))
                peaks = fit.find_peaks(smoothed, width=peak_width)[0]

                plt.plot(x_range, smoothed)
                plt.scatter(peaks, smoothed[peaks], c='r')
                plt.show()

                if preprocess.get_bool("Satisfied?"):
                    break



            while True:
                
                fitted_peaks = []

                half_width = int(input("Half width for fitting: "))

                # Estimate center of peaks and get uncertainty. Save them.

                for peak in peaks:

                    intensity_errors = float(fit.get_intensity_errors(intensities[peak-half_width:peak+half_width+1], smoothed[peak-half_width:peak+half_width+1], plot=False))
                    debug(f"{intensity_errors=}")

                    a, mu, sig, offset, mu_std = fit.find_single_peak(intensities[peak-half_width:peak+half_width+1], intensity_errors)

                    xg = range(2*half_width + 1)
                    values = fit.gaussian_func(xg, a, mu, sig, offset)
                    plt.plot(x_range, smoothed)
                    plt.scatter(x_range, intensities, s=0.3)
                    plt.plot(xg + peak - half_width, values, c='r')
                    plt.axvline(x=(mu + peak - half_width), c='r')
                    plt.axvline(x=(mu + peak - half_width) - mu_std, c='r', linestyle='--')
                    plt.axvline(x=(mu + peak - half_width) + mu_std, c='r', linestyle='--')
                    plt.show()

                    if preprocess.get_bool("ok?"):
                        fitted_peaks.append((mu + peak - half_width, mu_std))

                plt.plot(x_range, smoothed)

                for peak, err in fitted_peaks:
                    plt.axvline(x=peak)

                plt.show()

                if preprocess.get_bool("Satisfied?"):
                    peak_save_path = f"{save_path}/{filename[:-4]}_peaks.txt"
                    with open(peak_save_path, "w") as f:
                        f.write('\n'.join([f"{val[0]}, {val[1]}" for val in fitted_peaks]))
                    break

    if args.combine:
        # Load the lines peaks into a DF
        df = combine.load_line(save_path, element, line, 'intensity_peaks.txt')
        k_df = combine.compute_k(df, f'{save_path}/{element}_{line}_k.csv')
    else:
        k_df = pd.read_csv(f'{save_path}/{element}_{line}_k.csv')
    
    if args.fit_graph:
        ...
    
    