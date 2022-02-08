#!/usr/bin/env python
from XinputFile import XinputFile as xif
from Xmendeleev import Xmendeleev as xm
import os
import subprocess
import numpy as np
import time
import datetime

def do_process(input_file, ncpu = None, poisson = False):
    if not ncpu:
        ncpu = os.cpu_count() if os.cpu_count() <= 4 else os.cpu_count() -2
    if os.name == 'posix':
        command = ["xmimsim", "--set-threads", str(ncpu), "--disable-gpu"]
    else:
        command = ["C:\\Program Files\\XMI-MSIM 64-bit\\Bin\\xmimsim-cli.exe",
                   "--set-threads", str(ncpu),
                   "--disable-gpu"]
    if poisson:
        command += ["--enable-poisson"]
    print(f"processing {input_file} with {ncpu} cores")
    p = subprocess.Popen(command + [input_file])
    p.wait()

def weight_fraction_combinations(wfracsteps, num_elements):
    wfrac = [np.linspace(0,1,wfracsteps) for _ in range(num_elements)]
    if num_elements == 1:
        return np.round(wfrac[0] * 100, 2)
    w_fraction = np.array(np.meshgrid(*wfrac))
    w_fraction = w_fraction.T.reshape(-1,num_elements)[1:]
    #normalize w_fraction
    w_fraction_sum = w_fraction.sum(axis = 1)
    for c in range(w_fraction.shape[1]):
        w_fraction[:,c] = 100 * w_fraction[:,c]/w_fraction_sum
    #round
    w_fraction = np.round(w_fraction, 2)
    return np.unique(w_fraction, axis = 0)

def random_weight_fraction(num_elements, N = 1):
    rwf_list = [np.random.rand((num_elements)) for _ in range(N)]
    rwf = np.vstack(rwf_list)
    #normalize
    rwf_sum = rwf.sum(axis = 1)
    for c in range(rwf.shape[1]):
        rwf[:,c] = 100 * rwf[:,c]/rwf_sum
    #round
    rwf = np.round(rwf, 2)
    return np.unique(rwf, axis = 0)

def read_elements(file_to_read : str, xm_inst) -> list :
    if not isinstance(file_to_read, str):
        raise TypeError(f'String expected: {file_to_read}')
    elements_list = list()
    with open(file_to_read) as f2r:
        for line in f2r:
            if line.strip()[0] == '#':
                continue
            #print(f"loading  {_:12} data")
            elements = [xm_inst.get_element(elem) for elem in line[:-1].split(',')]
            elements_list.append(elements)
    return elements_list

def random_weight_fraction2(num_elements, N = 1):
    if num_elements < 2:
        raise ValueError("expected more than one elements")
    rwf = np.random.rand(N, num_elements - 1) * 100
    rwf = np.concatenate([np.zeros((N,1)), rwf, np.array([100]*N).reshape(N,1)], axis = 1)
    rwf.sort(axis = 1)
    out = np.empty((N, num_elements))
    for c in range(rwf.shape[1] - 1):
        out[:,c] = rwf[:, c + 1] - rwf[:, c]
    return np.unique(np.round(out,2), axis = 0)

if __name__ == "__main__":
    print(time.ctime())
    ifile = xif()
    ifile.live_time = 0.3 * 0.35
    slit_size = 0.001
    ifile.slit_size_x = slit_size
    ifile.slit_size_y = slit_size
    #ifile.hydrocerussite_thickness = 0.0035
    ifile.d_sample_detector = 1.5
    lt = time.localtime()
    timestamp = f'{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday}_{lt.tm_hour}:{lt.tm_min}:{lt.tm_sec}'
    ifile.set_root_dir(f'/home/rosario/xmimsim/syntetic/{timestamp}')
    #ifile.set_ref_lay_thickness(0.00015)
    #elements_list = read_elements(os.path.join(root_dir,'xsimspe','elements.txt'),xm())
    ref_lay_thickness_upper_bound =  4.5e-05
    hydro_thickness_bounds= [0.0015, 0.004]
    #loop on reference layer elements
    #for ref_lay_elements in elements_list:
        #ifile.ref_lay_elements = ref_lay_elements
    ifile.add_ref_layer_element('Cu')
    ifile.add_ref_layer_element('Zn')
    #ifile.add_ref_layer_element('Ti')
    #loop on weight fractions
    #elem_num = len(ifile.ref_lay_elements)
    #wfractions = random_weight_fraction2(elem_num, 100)
    #wfrac_steps = 5 #if elem_num == 2 else 4
    #wfractions = weight_fraction_combinations(wfrac_steps, elem_num)
    #for rlthick in np.linspace(0.00001, 0.001, 10):
        #ifile.ref_lay_thickness = round(rlthick, 6)
    #for wf in wfractions:
    for i in range(100):
        Cuwf = round(90 + np.random.rand() * 10, 2)
        Znwf = round(100 - Cuwf, 2)
        ifile.ref_lay_wfractions = np.array([Cuwf, Znwf])
        ifile.hydrocerussite_thickness = round(hydro_thickness_bounds[0] + np.random.rand() * (hydro_thickness_bounds[1] - hydro_thickness_bounds[0]), 6)
        ifile.ref_lay_thickness = round(np.random.rand() * ref_lay_thickness_upper_bound, 6)
        input_file = ifile.gen_input_file()
        start = time.time()
        do_process(input_file)
        stop = time.time()
        print(f"enlapsed time: {datetime.timedelta(seconds = stop - start)}")
    
