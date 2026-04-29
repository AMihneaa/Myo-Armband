from config import BASE_STAT

def iter_windows(N, win, hop):
    i = 0
    while i + win <= N:
        yield i, i + win
        i += hop

def make_feature_names(n_channels, bp_bands, ievd_k):
    names = []

    for ch in range(n_channels):
        for nm in BASE_STAT:
            names.append(f"ch{ch}_{nm}")

    for ch in range(n_channels):
        for (lo, hi) in bp_bands:
            lo_s = str(lo).replace(".", "p")
            hi_s = str(hi).replace(".", "p")
            names.append(f"ch{ch}_bp_{lo_s}_{hi_s}")
            names.append(f"ch{ch}_bpR_{lo_s}_{hi_s}")

    for ch in range(n_channels):
        for i in range(ievd_k):
            names.append(f"ch{ch}_ievd_s{i+1}")

    return names