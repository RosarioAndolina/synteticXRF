#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import os
sys.path.append('/home/rosario/xmimsim/xsimspe')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, SpanSelector, CheckButtons, Button
import matplotlib.colors as mcolors
import spectra_utils as su
from collections import namedtuple
from joblib import Parallel, delayed
from scipy.optimize import minimize
import pandas as pd

usage = """
Usage ./inspect_data.py [-d | --datadir] [-h | --help]

Options:

-d, --datadir     Outdata directory, usually /some_path/outdata
    --timax       Time multiplication factor max value
-h, --help        Print this message and exit successfully

"""

datadir = ""
tmax = 2

index = 1
while index < len(sys.argv):
    if sys.argv[index] in ["--datadir", "-d"]:
        datadir = sys.argv[index + 1]
        index += 1
    elif sys.argv[index] in ["-h", "--help"]:
        print(usage)
        sys.exit(0)
    elif sys.argv[index] == "--tmax":
        tmax = int(sys.argv[index +1])
        index += 1
    else:
        raise ValueError(f"Unrecognized option {sys.argv[index]}")
    index += 1
if not datadir:
    raise ValueError("Missing required option --datadir or -d")

#datadir = "/home/rosario/xmimsim/test/rlthickness/outdata"
dirlist = os.listdir(datadir)
data = dict()
spectrum = namedtuple("spectrum", ["counts", "info"])
xrdata = pd.read_csv('Xraydata.csv')
COLORS = list(mcolors.TABLEAU_COLORS)
LINESFLAG = True
CALIBFLAG = False

def process_spectrum(fname):
    counts = su.load_xmso(fname)[0]
    info = su.parse_xmso_fname(os.path.basename(fname))
    s = spectrum(counts = counts, info = info)
    return s

def process_rand(fname):
    counts = su.load_xmso(fname)[0]
    info = su.parse_xmso_fname(os.path.basename(fname))
    counts = su.get_randomMC(counts, RANDOMCOUNTS)
    s = spectrum(counts = counts, info = info)
    return s

def get_spectrum(target_dir, data_dict, process_func):
    fnames = os.listdir(os.path.join(datadir, target_dir))
    spectra = Parallel(n_jobs = os.cpu_count())(delayed(process_func)(os.path.join(datadir, target_dir, xmso_file)) for xmso_file in fnames)
    data_dict[target_dir] = spectra

#read the data
for _dir in dirlist:
    get_spectrum(_dir, data, process_spectrum)

max_len = 0
for values in data.values():
    max_len = max(max_len, len(values))
    
data["energy"] = su.load_xmso(f"{datadir}/{_dir}/" + os.listdir(os.path.join(datadir, _dir))[0])[1]
# data["energy"] = su.load_xmso(os.path.join(datadir,
                                           # "Ca",
                                           # "xsimspe_Ca-66.67_0.00223_hydro0.0035_live-time0.2_d-samp-det1.5_d-samp-src6_slit0.001_photons100000.xmso"))[1]


def poli1(x, p0, p1):
        return p0 + p1*x
    
def chi2(par, x, y):
    ye = poli1(x, *par)
    return np.sum(((y-ye)**2)/ye)

def lsqLoss(par, x, y):
    ye = poli1(x, *par)
    return np.sum((y - ye)**2)

def calibrate(calib_ini):
    data_x, data_y = list(), list()
    with open(calib_ini) as ini:
        for line in ini:
            _x, _y = line.split()
            print(_x, _y)
            data_x.append(float(_x))
            data_y.append(float(_y))
            
    iguess = [0, (data_y[1] - data_y[0]) / (data_x[1] - data_x[0])]
    return  minimize(chi2, iguess, args = (np.array(data_x), np.array(data_y)))

xrfdatadir = '/home/rosario/xrfxrdData/ElGreco_XRF'
elgreco = np.load(os.path.join(xrfdatadir, 'data.npy'))
elgreco = elgreco.reshape(elgreco.shape[0]*elgreco.shape[1], elgreco.shape[2])
result = calibrate(os.path.join(xrfdatadir, 'Edf', 'calibration.ini'))
global energy
energy = result.x[0] + result.x[1] * np.arange(elgreco.shape[1])
xrf_peaks_list = list()
synt_peaks_list = list()

def plot_data():
    elem = list(data.keys())
    spec_index = 0
    elem_index = 0
    fig, ax = plt.subplots()
    fig.set_size_inches(15,8)
    line1, = ax.plot(data["energy"], data[elem[elem_index]][spec_index].counts, label = data[elem[elem_index]][spec_index].info.elements)
    line2, = ax.plot(energy,elgreco[0], label = 'elgreco')
    ax.set_xlabel('energy (keV)')
    ax.set_ylabel('counts')
    ax.set_ylim(0, np.max(elgreco))
    L = plt.legend()
    max_counts = 0

    plt.subplots_adjust(left = 0.3, bottom = 0.25)
    axspec = plt.axes([0.25, 0.1, 0.65, 0.03])
    spec_slider = Slider(ax = axspec,
                         label = "spectra",
                         valmin = 0,
                         valmax = max_len,
                         valinit = 0,
                         valstep = 1)

    axelem = plt.axes([0.1, 0.25, 0.0225, 0.63])
    elem_slider = Slider(ax = axelem,
                         label = "Ba-Ti",
                         valmin = 0,
                         valmax = len(elem) -2,
                         valinit = 0,
                         valstep = 1,
                         orientation = "vertical")
    
    axXRF = plt.axes([0.15, 0.25, 0.0225, 0.63])
    XRF_slider = Slider(ax = axXRF,
                         label = "ElGreco",
                         valmin = 0,
                         valmax = elgreco.shape[0],
                         valinit = 0,
                         valstep = 1,
                         orientation = "vertical")
                         
    axtime = plt.axes([0.25, 0.15, 0.65, 0.03])
    time_slider = Slider(ax = axtime,
                         label = "time_correction",
                         valmin = 0,
                         valmax = tmax,
                         valinit = 1,
                         valstep = 0.01)
                         
    axckbtn = plt.axes([0.2, 0.65, 0.05, 0.1])
    check = CheckButtons(axckbtn, ["recalibrate"], [False])
    axckbtn.axis('off')
    
    def onselect(emin, emax):
        if check.get_status()[0] == True:
            condition = lambda x, m, M : (x > m) & (x < M)
            xrf_peak = np.average(energy[condition(energy,emin,emax)], weights = elgreco[XRF_slider.val, condition(energy,emin,emax)])
            xrf_peaks_list.append(xrf_peak)
            weights = data[elem[elem_slider.val]][spec_slider.val].counts
            synt_peak = np.average(data["energy"][condition(data["energy"],emin,emax)],
                                   weights = weights[condition(data["energy"],emin,emax)])
            synt_peaks_list.append(synt_peak)
            return
        for c in xrdata.columns[2:5]:
            results = xrdata.loc[(xrdata[c] > emin * 1000) & (xrdata[c] < emax * 1000), ["Symbol", c]]
            label = " ".join(["-".join([str(x), str(y)]) for x in results.Symbol.values for y in results.columns.values[1:]])
            evalues = results.iloc[:, 1].values/1000
            ax.vlines(evalues, 0, np.max(elgreco), colors = COLORS[:len(evalues)], label = label)
            print(results)
        ax.legend()
        fig.canvas.draw_idle()
        #fig.canvas.flush_events()
    
    axbtn = plt.axes([0.2, 0.5, 0.05, 0.02])
    cal_btn = Button(axbtn, label = 'calibrate')
    
    def btn_func(event):
        print("xrf peaks", xrf_peaks_list)
        print("synt peaks", synt_peaks_list)
        iguess = [0, (synt_peaks_list[1] - synt_peaks_list[0]) / (xrf_peaks_list[1] - xrf_peaks_list[0])]
        print("iguess", iguess)
        result = minimize(lsqLoss, iguess, args = (np.array(xrf_peaks_list), np.array(synt_peaks_list)))
        if result.success:
            new_energy = poli1(energy, result.x[0], result.x[1])
            line2.set_xdata(new_energy)
            fig.canvas.draw_idle()
        else:
            print("Recalibrazion error")
    
    span = SpanSelector(ax,
                        onselect,
                        "horizontal",
                        useblit = True,
                        interactive = True,
                        drag_from_anywhere = True,
                        props = dict(alpha = 0.3, facecolor = "tab:blue"))
    
    
    def check_func(label):
        # if label == "lines":
            # LINESFLAG = False if LINESFLAG else True
            # CALIBFLAG = not L
        # else:
            # CALIBFLAG = True
            # LINESFLAG = False
        print(label)

    def update_xrf(val):
        #spectrum = elgreco[XRF_slider.val, :]
        ydata = elgreco[XRF_slider.val, :]
        line2.set_ydata(ydata)
        fig.canvas.draw_idle()
        
    def update(val):
        spectrum = data[elem[elem_slider.val]]
        if spec_slider.val < len(data[elem[elem_slider.val]]):
            info = data[elem[elem_slider.val]][spec_slider.val].info
            #print(data[elem[elem_slider.val]][spec_slider.val].info.elements)
            ydata = data[elem[elem_slider.val]][spec_slider.val].counts
            #line1.set_ydata(su.get_random2(ydata*10))
            line1.set_ydata(ydata * time_slider.val)
            L.get_texts()[0].set_text(f"{info.elements} rf_thickness: {info.reflay} hydro_thickness: {info.hydro}")
            elem_slider.label.set_text(elem[elem_slider.val])
            fig.canvas.draw_idle()

    elem_slider.on_changed(update)
    spec_slider.on_changed(update)
    time_slider.on_changed(update)
    XRF_slider.on_changed(update_xrf)
    check.on_clicked(check_func)
    cal_btn.on_clicked(btn_func)

    plt.show()

if __name__ == "__main__":
    plot_data()
