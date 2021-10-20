from math import ceil

import numpy as np
from scipy.signal import find_peaks, hilbert


def get_periods(serie: np.ndarray, time_step: float = 30 / 525600) -> np.ndarray:
    """
    Extract periods with most importance in 
    the Fourier Transformed Serie.

    Parameters
    ----------
    serie : np.ndarray
        The serie which you will analyze.

    Returns
    -------
    np.ndarray
        Periods with most importance in FFTed serie.

    """
    fft_serie = np.fft.fft(serie)
    fft_serie = fft_serie[:int(fft_serie.size/2)]
    fft_serie = hilbert(np.abs(fft_serie))
    peaks_idx = find_peaks(fft_serie)[0]
    peaks_idx = np.array(peaks_idx)
    periodos = np.round(1 / (peaks_idx * time_step))
    periodos = list(set(periodos))
    periodos.reverse()
    if 0 in periodos:
        periodos.remove(0)
    periodos = np.array(periodos, dtype=np.int64)[:10]
    return periodos


def dedecomp(serie: np.ndarray, to_subtract: np.ndarray, period: int) -> np.ndarray:
    comp = serie.copy()
    for i in range(len(serie)):
        comp[i] = np.mean(serie[i : period + i] - to_subtract[:, i].sum())
    return comp


def decomp(serie: np.ndarray, periods: np.ndarray) -> np.ndarray:
    """
    Decompose the serie according to the given periods.


    Parameters
    ----------
    serie : np.ndarray -> shape = (serie_size,)
        The serie which you will decompose.

    periods : np.ndarray
        Periods of reference

    Returns
    -------
    np.ndarray
        Decomposed Serie in #periods components

    """
    c = np.zeros((periods.shape[0], serie.shape[0]))
    for i in range(periods.shape[0]):
        c[i] = dedecomp(serie, c, periods[i])
    comp_dados = c.T
    return comp_dados
