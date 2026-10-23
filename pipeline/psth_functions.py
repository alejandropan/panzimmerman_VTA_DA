"""
Peri-event time histogram with a causal Gaussian kernel.

Adapted from the IBL `brainbox.singlecell.calculate_peths`: the kernel is
truncated to its trailing half, so a bin's rate depends only on spikes at or
before it.
"""

# %%
import numpy as np
from iblutil.util import Bunch
from scipy.signal import convolve
from scipy.signal.windows import gaussian


def calculate_causal_peths(spike_times, spike_clusters, cluster_ids,
                           align_times, pre_time=0.2, post_time=0.5,
                           bin_size=0.01, smoothing=0.0255, return_fr=True):
    """
    Peri-event time histograms, mean and standard deviation across events.

    Args:
        spike_times: Spike times in seconds.
        spike_clusters: Cluster id of each spike.
        cluster_ids: Clusters to compute.
        align_times: Event times in seconds.
        pre_time: Seconds before each event.
        post_time: Seconds after each event.
        bin_size: Bin width in seconds.
        smoothing: Standard deviation in seconds of the causal Gaussian; 0 skips
            smoothing.
        return_fr: Firing rate in Hz if True, spike counts if False.

    Returns:
        `(peths, binned_spikes)`. `peths` is a Bunch with `means`, `stds`,
        `tscale` and `cscale`; `binned_spikes` is events x clusters x bins.
    """
    n_offset = 5 * int(np.ceil(smoothing / bin_size))
    n_bins_pre = int(np.ceil(pre_time / bin_size)) + n_offset
    n_bins_post = int(np.ceil(post_time / bin_size)) + n_offset
    n_bins = n_bins_pre + n_bins_post
    binned_spikes = np.zeros(shape=(len(align_times), len(cluster_ids), n_bins))

    if smoothing > 0:
        w = n_bins - 1 if n_bins % 2 == 0 else n_bins
        window = gaussian(w, std=smoothing / bin_size)
        window[:int(np.ceil(w / 2))] = 0
        window /= np.sum(window)
        binned_spikes_conv = np.copy(binned_spikes)

    ids = np.unique(cluster_ids)

    idxs = np.bitwise_and(spike_times >= np.min(align_times) - (n_bins_pre + 1) * bin_size,
                          spike_times <= np.max(align_times) + (n_bins_post + 1) * bin_size)
    idxs = np.bitwise_and(idxs, np.isin(spike_clusters, cluster_ids))
    spike_times = spike_times[idxs]
    spike_clusters = spike_clusters[idxs]

    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    for i, t_0 in enumerate(align_times):
        ts = tscale + t_0
        idxs = np.bitwise_and(spike_times >= ts[0], spike_times <= ts[-1])
        i_spikes = spike_times[idxs]
        i_clusters = spike_clusters[idxs]

        xscale = ts
        xind = (np.floor((i_spikes - np.min(ts)) / bin_size)).astype(np.int64)
        yscale, yind = np.unique(i_clusters, return_inverse=True)
        nx, ny = [xscale.size, yscale.size]
        ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
        r = np.bincount(ind2d, minlength=nx * ny, weights=None).reshape(ny, nx)

        bs_idxs = np.isin(ids, yscale)
        binned_spikes[i, bs_idxs, :] = r[:, :-1]

        if smoothing > 0:
            idxs = np.where(bs_idxs)[0]
            for j in range(r.shape[0]):
                binned_spikes_conv[i, idxs[j], :] = convolve(
                    r[j, :], window, mode='same', method='auto')[:-1]

    if smoothing > 0:
        binned_spikes_ = np.copy(binned_spikes_conv)
    else:
        binned_spikes_ = np.copy(binned_spikes)
    if return_fr:
        binned_spikes_ /= bin_size

    peth_means = np.mean(binned_spikes_, axis=0)
    peth_stds = np.std(binned_spikes_, axis=0)

    if smoothing > 0:
        peth_means = peth_means[:, n_offset:-n_offset]
        peth_stds = peth_stds[:, n_offset:-n_offset]
        binned_spikes = binned_spikes[:, :, n_offset:-n_offset]
        tscale = tscale[n_offset:-n_offset]

    tscale = (tscale[:-1] + tscale[1:]) / 2
    peths = Bunch({'means': peth_means, 'stds': peth_stds, 'tscale': tscale,
                   'cscale': ids})
    return peths, binned_spikes
