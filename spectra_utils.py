#!/usr/bin/env python

import xml.etree.ElementTree as et
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import sys
import glob
import os
from csaps import csaps
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from XRDXRFutils import DataXRF


def gen_MPhistogram(xml_spectrum, parsed_file, q = None):
    spectrum_conv = xml_spectrum.find("spectrum_conv")
    nchannels = len(spectrum_conv.findall("channel"))
    chnum = np.empty((nchannels), dtype = np.uint)
    energy = np.empty((nchannels))
    counts = np.empty((nchannels))
    for channel in spectrum_conv.findall("channel"):
        channelnr = int(channel.find("channelnr").text)
        chnum[channelnr] = channelnr
        energy[channelnr] = float(channel.find("energy").text)
        for count in channel.findall("counts"):
            if count.attrib["interaction_number"] == "2":
                counts[channelnr] = float(count.text)
    channel_hist = np.histogram(chnum, bins = nchannels, weights = counts)
    energy_hist = np.histogram(energy, bins = nchannels, weights = counts)
    hists = namedtuple("histogram", ["channel", "energy", "info"])
    retval = hists(channel = channel_hist,
                   energy = energy_hist,
                   info = parsed_file)
    if q:
        q.put(retval)
    else:
        return retval

def load_xmso(xmso_file, interaction_number = 2):
    interaction_number = str(interaction_number)
    #parsed_file = parse_file_name(xmso_file)
    xml_spectrum = et.parse(xmso_file)
    spectrum_conv = xml_spectrum.find("spectrum_conv")
    nchannels = len(spectrum_conv.findall("channel"))
    chnum = np.empty((nchannels), dtype = np.uint)
    energy = np.empty((nchannels))
    counts = np.empty((nchannels))
    for channel in spectrum_conv.findall("channel"):
        channelnr = int(channel.find("channelnr").text)
        chnum[channelnr] = channelnr
        energy[channelnr] = float(channel.find("energy").text)
        for count in channel.findall("counts"):
            if count.attrib["interaction_number"] == interaction_number:
                counts[channelnr] = float(count.text)
    return (counts, energy)

def parse_xmso_fname(xmso_fname):
    info = namedtuple("simpar", ["elements","wf","reflay","hydro","live_time","d_samp_det","slit","n_photons"])
    xmso_fname = os.path.basename(xmso_fname)
    xmso_fname = xmso_fname.split('_')
    elements = xmso_fname[1]
    elements = elements.split('-')
    wf = [float(x) for i, x in enumerate(elements) if (i % 2) == 1]
    #wf = [float(elements[1]), float(elements[3])]
    out = info(elements = xmso_fname[1],
               wf = wf,
               reflay = float(xmso_fname[2]),
               hydro = float(xmso_fname[3][5:]),
               live_time = float(xmso_fname[4][9:]),
               d_samp_det = float(xmso_fname[5][10:]),
               slit = float(xmso_fname[7][4:]), 
               n_photons = int(xmso_fname[8][7:-5]))
    return out

def get_random(counts, bins, num = 10000):
    #counts, bins = hist.energy
    # normalize counts
    snormal = counts/np.sum(counts)
    # compiute CDF
    cdf = np.empty((counts.shape[0]))
    cdf[0] = 0 #snormal[0]
    for i in range(1,snormal.shape[0]):
        #cdf[i] = np.sum(snormal[: i + 1])
        cdf[i] = np.sum(snormal[: i])
    cdf[-1] = 1
    #interpolate the CDF
    interp_inverse_cdf = interp1d(cdf, bins[:-1], kind = 'linear')
    # get randoms
    out = np.empty((num))
    for i in range(num):
        r = np.random.rand()
        #if r < cdf[0]:
        #    continue
        out[i] = interp_inverse_cdf(r)
    hout = np.histogram(out, bins = bins)
    return hout

def get_random2(counts):
    rounded = np.floor(counts)
    err = counts - rounded
    r = np.random.rand((counts.shape[0]))
    rounded += np.int32(r <= err)
    return np.random.poisson(rounded)

def get_random3(counts, num = 2000, poisson = False):
    norm_counts = counts/np.sum(counts)
    one_pixel_spec = np.zeros_like(counts)
    for _ in range(num):
        r = np.random.rand((counts.shape[0]))
        one_pixel_spec += np.int32(r <= norm_counts)
    if poisson:
        return np.random.poisson(one_pixel_spec)
    return one_pixel_spec

def get_randomMC(counts, N = 1500):
    norm_counts = counts/np.sum(counts)
    r = np.random.rand(N, counts.shape[0])
    out = r <= norm_counts
    return out.sum(axis = 0)    

def xmso_to_hist(xmso_file):
    parsed_file = parse_file_name(xmso_file)
    xml_file = et.parse(xmso_file)
    hist = gen_MPhistogram(xml_file, parsed_file)
    return hist


def xmso_to_sigle_pixel(xmso_file, rand_func = get_random2, *args):
    counts, energy = load_xmso(xmso_file, interaction_number = 2)
    return rand_func(*args)
    

def load_xrfdata(datadir, calibrate = True):
    data = DataXRF()
    if calibrate:
        data.calibrate_from_file(os.path.join(datadir, 'calibration.ini'))
        data.read(datadir)
        data.save_h5()
    data.load_h5(os.path.join(datadir, 'data.h5'))
    return data

def load_xraydata(data_fname = "/home/rosario/xmimsim/xsimspe/Xraydata.csv"):
    pass

def show_spectra(spectra, **kwargs):
    if not isinstance(spectra, list):
        spectra = [spectra]
    x = np.arange(0,spectra[0].shape[0],1)
    with plt.ion():
        fig = plt.figure()
        plt.show()
        for i, s in enumerate(spectra):
            plt.title(f'Spectrum No {i+1}')
            plt.plot(x, s)
            _ = input()
            if "cla" in kwargs.keys() and kwargs["cla"] == True:
                plt.cla()
        

def baseline(counts, filter_size, filter_method = None):
    fondo = np.empty((len(counts)))
    if filter_size % 2 == 0:
        filter_size += 1
    padding_len = filter_size // 2
    padding = np.zeros((padding_len))
    padded = np.concatenate((padding, counts, padding))
    index = 0
    pivot = index + padding_len
    while index < len(counts):
        pivot_value = padded[pivot]
        filter_left = padded[index: pivot]
        filter_right = padded[pivot + 1 : pivot + padding_len + 1]
        _filter = (filter_left + filter_right)/2.
        #_filter = np.append(_filter,[pivot_value])
        if filter_method == 'mean':
            fondo[index] = np.mean(_filter)
        elif filter_method == 'median':
            fondo[index] = np.median(_filter)
        else:
            fondo[index] = np.min(_filter)
        index += 1
        pivot += 1
    return fondo

def Gauss(x, A, m, s):
    return A/(s*np.sqrt(2*np.pi))*np.exp(-0.5*((x-m)/s)**2)

def Lorentzian(x, A, x0, gamma):
    return A*gamma*0.5/((x - x0)**2 + (gamma*0.5)**2)/np.pi

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def mean_filter(data, kernel_size):
    out = data.copy()
    if kernel_size % 2 == 0:
        kernel_size += 1
    plen = kernel_size // 2
    padding = np.zeros((plen))
    p = np.concatenate((padding, data, padding))
    index = 0
    while index < len(data):
        mean = np.mean(p[index:index + kernel_size])
        out[index] = mean
        index += 1
    return out

def Baseline(data, kernel_size = 33, inc_size = 53, filter_method = 'min'):
    if inc_size % 2 == 0:
        inc_size += 1
    out = data.copy()
    halfk = kernel_size // 2
    pad = max(halfk // 2, inc_size)
    padding = np.zeros((pad))
    x = np.concatenate((padding, data, padding))
    index = 0
    pivot = index + pad
    counts_max = np.max(data)
    ksize = np.empty((data.shape[0]))
    while index < len(data):
        D = max(0, np.abs((x[pivot + inc_size] - x[pivot])/counts_max))
        Dhalfk = max(pad,int(halfk * D))
        ksize[index] = Dhalfk
        #print(Dhalfk)
        filter_left = x[pivot - Dhalfk : pivot]
        filter_right = x[pivot + 1 : pivot + Dhalfk + 1]
        _filter = np.append((filter_left + filter_right)/2., x[pivot])
        #_filter = np.append(_filter,[pivot_value])
        if filter_method == 'mean':
            out[index] = np.mean(_filter)
        elif filter_method == 'median':
            out[index] = np.median(_filter)
        else:
            out[index] = np.min(_filter)
        index += 1
        pivot += 1
    return (out, ksize)

def Derifilter(data, inc_size = 51):
    out = data.copy()
    padding = np.zeros((inc_size))
    x = np.concatenate((padding,data,padding))
    index = 0
    max_counts = np.max(data)
    while index < len(data):
        D = max(0, x[index + 2*inc_size] - x[index])
        out[index] = D/max_counts
        index +=  1
    return out

def Continuum(data, inc_size = 51):
    data_norm = data/np.max(data)
    data_filt = Derifilter(data_norm, inc_size)
    data_peaks, _ = find_peaks(data_norm, height = 0.7)
    data_filt_peaks, _ = find_peaks(data_filt, height = 0.7)
    pad = data_peaks[0] - data_filt_peaks[0]
    print(f'pad: {pad}')
    out = np.concatenate((np.zeros((pad)), data_filt))
    return out[:-pad]

if __name__ == "__main__":
    workdir = "/home/rosario/progetti/xmimsim/scripts/xsimspe"
    datadir = "/home/rosario/progetti/xmimsim/outdata"
    #datadirs = {e_dir : os.path.join(datadir, e_dir) for e_dir in ["Co-Fe","Cu-Zn","Fe-Mn"]}
    fname = os.path.join(datadir, "Co-Fe", "xsimspe_Co33.33_Fe66.67_0.002_hydrocerussite.xmso")
    hist = xmso_to_hist(fname)
    counts, bins = hist.energy
    filter_size = 50
    #fondo = baseline(counts, 80,'min')
    #fondo = baseline(fondo, 50, 'median')
    #fondo = baseline(fondo, 80, 'min')
    #fondo_spline = csaps(bins[:-1],fondo, bins[:-1], smooth = 0.992)
    spline = csaps(bins[:-1],counts, bins[:-1], smooth = 0.99)
    plt.figure(1)
    #plt.hist(bins[:-1], bins = bins, weights = fondo_spline, log = True, histtype = 'step', label = 'spline')
    #plt.hist(bins[:-1], bins = bins, weights = fondo, log = True, histtype = 'step', label = 'spline')
    plt.hist(bins[:-1], bins = bins, weights = counts, color = 'black', log = True, histtype = 'step')
    plt.hist(bins[:-1], bins = bins, weights = spline, log = True, histtype = 'step', label = 'spline')
    plt.legend()
    plt.show()
    sys.exit()
    index = 0
    filter_size = 100
    if filter_size % 2 == 0:
        filter_size += 1
    padding = np.zeros((filter_size // 2))
    padding_len = len(padding)
    fondo_mean = np.empty((len(fondo)))
    padded_counts = np.concatenate((padding, fondo, padding))
    while index < len(counts):
        _filter = padded_counts[index : index + filter_size]
        fondo_mean[index] = np.mean(_filter)
        index += 1
    plt.hist(bins[:-1], bins = bins, weights = fondo_mean, log = True, histtype = 'step', label = f'window size: {filter_size}')
    plt.title(hist.info.elements)
    plt.legend()
    plt.figure(2)
    diff = counts - fondo_mean
    diff[diff < 0] = 0
    plt.hist(bins[:-1], bins = bins, weights = diff, histtype = 'step')
    plt.yscale('symlog')
    plt.show()
