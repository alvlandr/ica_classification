import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nilearn import image, plotting
from nilearn.masking import apply_mask, compute_epi_mask


class ICReader:
    """
    Initializing base fMRI independent components entities: image, raw data, time series, Fourier transformation.
    Currently it is assumed that input data is the output of MELODIC.
    """
    def __init__(self, ic_path, ic_ts_path=None, ic_ft_path=None):
        self.ic_path = ic_path
        self.ic_ts_path = ic_ts_path
        self.ic_ft_path = ic_ft_path
        self.ic_dir = os.path.dirname(ic_path)
        self.TR = 0.7

    def run(self):
        self.image = image.load_img(self.ic_path)
        self.raw_data = self.image.get_fdata()

        if self.ic_ts_path is not None:
            self.timeseries = pd.read_csv(self.ic_ts_path, sep=' ', header=None).dropna(axis=1).values
        else:
            mask_img = compute_epi_mask(self.image,0,1)
            self.timeseries = apply_mask(self.image, mask_img)

        if self.ic_ft_path is not None:
            self.ftransorm = pd.read_csv(self.ic_ft_path, sep=' ', header=None).dropna(axis=1).values
        else:
            # TODO: implement correct FT
            print("Fourier transformation calculation is not supported now.")
            self.ftransorm = np.abs(np.fft.rfft(self.timeseries))

    def plot_spatial(self, ic_num):
        ic_cur = image.index_img(self.image, ic_num)
        # TODO: add background structural
        print("No background provided. Using default.")
        plotting.plot_stat_map(ic_cur, threshold=2, title="IC {}".format(ic_num), display_mode="z",
                               cut_coords=(1, 10, 30))
        plotting.show()

    def plot_time_series(self, ic_num):
        plt.plot(self.timeseries[:, ic_num])
        plt.show()

    def plot_power_spectrum(self, ic_num):
        plt.plot(np.array(range(len(ic.ftransorm[:, ic_num])))/self.timeseries.shape[0], self.ftransorm[:, ic_num])
        plt.show()


if __name__ == '__main__':
    ic = ICReader(
        ic_path=r'C:\Users\institute1\PycharmProjects\ica_classification\resource\samples\HCP_hp2000\1.ica\filtered_func_data.ica\melodic_IC.nii.gz',
        ic_ts_path=r'C:\Users\institute1\PycharmProjects\ica_classification\resource\samples\HCP_hp2000\1.ica\filtered_func_data.ica\melodic_mix',
        ic_ft_path=r'C:\Users\institute1\PycharmProjects\ica_classification\resource\samples\HCP_hp2000\1.ica\filtered_func_data.ica\melodic_FTmix')
    ic.run()
    print(ic)
