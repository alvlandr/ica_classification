import numpy as np
from scipy import stats
from statsmodels.tsa.ar_model import AR


class FeatureGenerator:
    def __init__(self, ic):
        self.ic = ic

    def _generate_time_features(self):
        def _feature1():
            """ The number of independent components, as determined by MELODIC """
            return self.ic.raw_data.shape[-1]

        def _feature_AR(_ts):
            """ Features 2-8 related to AR model of the signal ts """
            ar1 = AR(_ts).fit(maxlag=1)
            ar2 = AR(_ts).fit(maxlag=1)

            def _feature_2_3(ts=_ts):
                """
                The relationship between the order of the AR model and its goodness of fit.

                It is interpreted in the following way: for models AR(1), AR(2), ... , AR(6) calculates mean residual
                and calculate linear regression versus [1, 2, ..., 6]. The result is coefficient for the regression.
                """
                ar_results = np.empty(6)
                orders = np.array(range(1, 7))
                for i in range(6):
                    ar_results = AR(ts).fit(maxlag=i).resid.mean()
                slope, _intercept = stats.linregress(orders, ar_results)

                return slope

            def _feature_4_5(model=ar1):
                """ The parameter and the residual of AR(1) """
                c_1_1 = model.params['L1.y']
                nu_1 = model.resid.var()
                return c_1_1, nu_1

            def _feature_6_8(model=ar2):
                """ The parameters and the residual of AR(2) """
                c_2_1 = model.params['L1.y']
                c_2_2 = model.params['L2.y']
                nu_2 = model.resid.var()
                return c_2_1, c_2_2, nu_2

            return _feature_2_3(_ts), _feature_4_5(model=ar1), _feature_6_8(model=ar2)

        def _feature_distributional(_ts):
            """ Timeseries ts distribution related features """
            def _feature_9_10(ts=_ts):
                """ The skewness and kurtosis of the time series """
                return stats.kurtosis(ts), stats.skew(ts)

            def _feature_11(ts=_ts):
                """ The difference between timeseries mean and its median """
                return np.mean(ts) - np.median(ts)

            def _feature_12_13(ts=_ts):
                """ Entropy (two different calculations): actually entropy and negentropy """
                p_ts = ts / ts.sum()
                entropy = -np.sum(p_ts * np.log2(ts))

                negentropy = np.mean(ts**3)**2 / 12 + stats.kurtosis(ts)**2

                return entropy, negentropy

            def _feature_14_19(ts=_ts):
                """ Timeseries ts jump characteristics """
                ts_diff = np.abs(np.diff(ts))

                def _exclude_largest_jumps(xts=ts, window=5):
                    """ Exlude largest jumps in every window """
                    ts_sub = []
                    for i in range(0, len(xts), window):
                        ts_tmp = set(np.abs(xts[i:i+window])) - {np.max(np.abs(xts[i:i+window]))}
                        ts_sub += list(ts_tmp)

                    return np.array(ts_sub)

                ts_sub = _exclude_largest_jumps(xts=ts, window=5)
                jump_1 = np.max(ts_diff) / np.std(ts)
                jump_2 = np.max(ts_diff) / np.std(ts_diff)
                jump_3 = np.mean(ts_diff) / np.std(ts)
                jump_4 = np.max(ts_diff) / np.mean(ts_sub)
                jump_5 = np.max(ts_diff) / np.sum(ts_sub)

                return jump_1, jump_2, jump_3, jump_4, jump_5

        def _feature_Fourier(_ft, _ts):
            """ Fourier transformation ft of timeseries ts related features """
            _freqs = np.array(range(_ft.shape[0])) / _ts.shape[0]

            def _feature_20_23(ft=_ft, freqs=_freqs):
                """
                The ratio of the sum of power above fHz to the sum of power below fHz, for f = 0.1, 0.15, 0.2 and 0.25
                """
                thresholds = [0.1, 0.15, 0.2, 0.25]
                results = [sum(ft[freqs >= threshold]) / sum(ft[freqs < threshold]) for threshold in thresholds]

                return results

            def _feature_24_30(ft=_ft, freqs=_freqs):
                """
                Percent of total power that falls in 0:0.01, 0.01:0.025, 0.025:0.05, 0.05:0.1, 0.1:0.15, 0.15:0.2 and
                0.2:0.25 Hz bins
                """
                bins = [[0, 0.01], [0.01, 0.025], [0.025, 0.05], [0.05, 0.1], [0.1, 0.15], [0.15, 0.2], [0.2, 0.25]]
                results = [sum(ft[(freqs >= _bin[0]) & (freqs < _bin[1])]) / sum(ft) for _bin in bins]

                return results

            def _feature_31_38(ft=_ft, ts=_ts, freqs=_freqs):
                """ Comparing the timeseries with their null model (i.e., convolving white noise with HRF) """
                bins = [[0, 0.01], [0.01, 0.025], [0.025, 0.05], [0.05, 0.1], [0.1, 0.15], [0.15, 0.2], [0.2, 0.25]]

                delta = 6 / self.ic.TR
                sigma = delta / 2
                generated_ft = np.empty([100, ts.shape[0]])
                for i in range(100):
                    generated_ft[i, :] = np.fft.hfft(
                        np.random.gamma(shape=(delta / sigma) ** 2, scale=delta / sigma ** 2, size=ts.shape[0]),
                        n=ft.shape[0])
                generated_ft_mean = generated_ft.mean(axis=0)

                results = \
                    [np.dot(
                        (ft[(freqs >= bin[0]) & (freqs < bin[1])]) - (generated_ft_mean[(freqs >= bin[0]) & (freqs < bin[1])]),
                        (ft[(freqs >= bin[0]) & (freqs < bin[1])]) - (generated_ft_mean[(freqs >= bin[0]) & (freqs < bin[1])]))
                     / np.dot(ft[(freqs >= bin[0]) & (freqs < bin[1])], ft[(freqs >= bin[0]) & (freqs < bin[1])]) ** 2
                     for bin in bins]
                results += [np.dot(ft, generated_ft_mean) / np.dot(ft,ft)]

                return results

        def _feature_correlations():
            pass



    def _generate_spatial_features(self):
        pass

    def run(self):
        pass
