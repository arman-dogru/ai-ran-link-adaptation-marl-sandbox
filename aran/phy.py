"""Toy PHY abstraction: a 15-entry CQI/MCS table, a logistic BLER curve per MCS, link budget helpers, Jain index.

The efficiencies are the LTE 4-bit CQI table (36.213 Table 7.2.3-1). The SINR thresholds are the commonly used
~10% BLER switching points for that table on AWGN; the logistic slope is a modelling choice (about 2.2 dB from 90%
to 10% BLER). This is an abstraction, not a link-level simulator."""
import numpy as np

EFF = np.array([0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141,
                2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547])  # bits/s/Hz per MCS index 0..14
THR10 = np.array([-6.7, -4.7, -2.3, 0.2, 2.4, 4.3, 5.9, 8.1,
                  10.3, 11.7, 14.1, 16.3, 18.7, 21.0, 22.7])  # dB SINR at which BLER = 10%
N_MCS = len(EFF)
SLOPE = 2.0  # 1/dB, logistic slope
LN9 = np.log(9.0)


def bler(sinr_db, mcs):
    """Block error probability of MCS index `mcs` at SINR `sinr_db` (broadcasts). BLER(THR10[m], m) = 0.1."""
    x = SLOPE * (np.asarray(sinr_db, float) - THR10[np.asarray(mcs)]) + LN9
    return 1.0 / (1.0 + np.exp(x))


def cqi_from_sinr(sinr_db):
    """Highest MCS index whose 10%-BLER threshold is <= SINR; -1 (CQI 0, out of range) below the lowest."""
    return np.searchsorted(THR10, np.asarray(sinr_db, float), side="right") - 1


def expected_goodput(sinr_db, mcs):
    return EFF[np.asarray(mcs)] * (1.0 - bler(sinr_db, mcs))


def best_mcs(sinr_db):
    """Genie MCS: argmax_m EFF[m] * (1 - BLER(sinr, m)) for each SINR (vectorised over the last axis = MCS)."""
    s = np.asarray(sinr_db, float)[..., None]
    return np.argmax(EFF * (1.0 - bler(s, np.arange(N_MCS))), axis=-1)


def pathloss_db(d_m):
    """3GPP TR 36.814 macro urban: 128.1 + 37.6 log10(d / km), d clipped at 35 m."""
    return 128.1 + 37.6 * np.log10(np.maximum(np.asarray(d_m, float), 35.0) / 1000.0)


def noise_dbm(bw_hz, nf_db):
    return -174.0 + 10.0 * np.log10(bw_hz) + nf_db


def db2lin(x):
    return 10.0 ** (np.asarray(x, float) / 10.0)


def lin2db(x):
    """dB of a linear power ratio; zero maps to -300 dB (no interference) instead of -inf."""
    return 10.0 * np.log10(np.maximum(np.asarray(x, float), 1e-30))


def jain(x):
    """Jain's fairness index (sum x)^2 / (n sum x^2), in [1/n, 1]; 1.0 for an all-zero vector by convention."""
    x = np.asarray(x, float)
    s2 = np.sum(x * x)
    return 1.0 if s2 == 0 else float(np.sum(x) ** 2 / (len(x) * s2))


def gauss_markov_rho(speed_kmh, fc_hz, tti_s):
    """Lag-1 correlation of a Clarke/Jakes fading process: J0(2 pi f_d T)."""
    from scipy.special import j0
    fd = speed_kmh / 3.6 * fc_hz / 299792458.0
    return float(j0(2 * np.pi * fd * tti_s))
