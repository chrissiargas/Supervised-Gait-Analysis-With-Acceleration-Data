from scipy.signal import find_peaks
import numpy as np

def find_peak_positions(prob_series):
    peaks, _ = find_peaks(prob_series, height=0.3, distance=10)
    return peaks

def refine_peak_positions(prob_series, candidate_peaks, window=10):
    refined = []
    times = []

    for peak in candidate_peaks:
        if peak not in times:
            start = max(0, peak - window)
            end = min(len(prob_series), peak + window)
            region = prob_series[start: end]
            times = np.arange(start, end)
            centroid = np.sum(times * region) / np.sum(region)
            refined.append(int(round(centroid)))
    return refined



