import Xmendeleev as xm
import os
import xml.etree.ElementTree as et
from xml.dom import minidom
import time
import numpy as np

class XinputFile():
    def __init__(self, input_template = None):
        if os.name == 'posix':
            home = os.getenv("HOME")
            self.root_dir = f'{home}/project/synteticXRF'
        else:
            self.root_dir = f'C:\\Users\XRAYLab\rosario-sim'
        self.work_dir = os.path.join(self.root_dir,'Xsim')
        self.input_files_dir = os.path.join(self.root_dir, 'input_files')
        self.ofname_is_set = False
        self.ifname_is_set = False
        os.makedirs(self.input_files_dir, exist_ok = True)
        self.data_dir = os.path.join(self.root_dir, 'outdata')
        os.makedirs(self.data_dir, exist_ok = True)
        if input_template:
            if os.path.exists(input_template):
                self.template = input_template
            elif os.path.exists(os.path.join(os.getcwd(), input_template)):
                self.template = os.path.join(os.getcwd(), input_template)
            else:
                raise FileNotFoundError(f"{input_template} not found")
        else:
            self.template = os.path.join(self.work_dir, 'input_template.xmsi.in')
        self.ref_lay_elements = []
        self.ref_lay_wfractions = np.array([])
        self.elem_data = xm.Xmendeleev()
        self.ref_lay_thickness = 0.002
        self.out_file_name = None
        self.n_photons_interval = 10000
        self.n_photons_line = 100000
        self.dtd_file = "http://www.xmi.UGent.be/xml/xmimsim-1.0.dtd"
        self.out_file_dirname = None
        self.input_file_name = None
        self.input_file_dirname = None
        self.hydrocerussite_thickness = 0.002
        self.slit_size_x = 0.05
        self.slit_size_y = 0.05
        self.live_time = 0.008
        self.nchannels = 2048
        self.gain = 0.0145656
        self.d_sample_detector = 1.5
        self.d_sample_source = 6
    
    def set_outdata_dir(self, outdata_dir):
            self.data_dir = outdata_dir
        
    def set_root_dir(self, root_dir):
        self.root_dir = root_dir
        self.work_dir = os.path.join(self.root_dir,'Xsim')
        self.input_files_dir = os.path.join(self.root_dir, 'input_files')
        os.makedirs(self.input_files_dir, exist_ok = True)
        self.data_dir = os.path.join(self.root_dir, 'outdata')
        os.makedirs(self.data_dir, exist_ok = True)
    
    def set_out_file_name(self, ofname):
        ex_index = ofname.find('.xmso')
        if ex_index != -1:
            ofname = ofname[:ex_index]
        basename = os.path.basename(ofname)
        dirname = os.path.dirname(ofname)
        self.out_file_dirname = dirname
        self.out_file_name = basename
        self.ofname_is_set = True
    
    def set_input_file_name(self, ifname):
        ex_index = ifname.find('.xmsi')
        if ex_index != -1:
            ifname = ifname[:ex_index]
        basename = os.path.basename(ifname)
        dirname = os.path.dirname(ifname)
        if dirname:
            self.input_file_dirname = dirname
        self.input_file_name = basename
        self.out_file_name = basename
        self.ifname_is_set = True
    
    def _unique_file(self, dirname, filename, fext):
        index = 1
        out = os.path.join(dirname, f"{filename}.{fext}")
        while os.path.exists(out):
            out = os.path.join(dirname, f"{filename}({index}).{fext}")
            index += 1
        return out
    
    def _layer_density(self):
        if len(self.ref_lay_elements) == 1:
            return round(self.ref_lay_elements[0].density, 5)
        return round(np.average([e.density for e in self.ref_lay_elements],
                           weights = self.ref_lay_wfractions), 5)
        
    
    def add_ref_layer_element(self, element, wfraction = 0):
        e = self.elem_data.get_element(element)
        self.ref_lay_elements.append(e)
        self.ref_lay_wfractions = np.append(self.ref_lay_wfractions, wfraction)
    
    def set_ref_lay_thickness(self, thickness):
        self.ref_lay_thickness = thickness
    
    def gen_input_file(self, poisson = False):
        if not self.ref_lay_elements:
            raise ValueError("reference layer is not defined")
        #normalize wfractions
        #self.ref_lay_wfractions = np.round(100 * self.ref_lay_wfractions/np.sum(self.ref_lay_wfractions), 2)
        lt = time.localtime()
        timestamp = f'{lt.tm_year}-{lt.tm_mon}-{lt.tm_mday} {lt.tm_hour}:{lt.tm_min}:{lt.tm_sec} (CET)'
        reference_layer = et.Element("layer")
        elements_symbols = []
        out_file_name = "xsimspe_" if not poisson else "xsimspe-poisson_"
        ewf = ""
        for elem, wfrac in zip(self.ref_lay_elements, self.ref_lay_wfractions):
            element = et.Element("element")
            reference_layer.append(element)
            atnum = et.SubElement(element, "atomic_number")
            atnum.text = str(elem.atomic_number)
            wfraction = et.SubElement(element, "weight_fraction")
            wfraction.text = str(wfrac)
            elements_symbols.append(elem.symbol)
            ewf += f"-{elem.symbol}-{wfrac}"
        out_file_name += ewf[1:]
        density = et.SubElement(reference_layer, "density")
        density.text = str(self._layer_density())
        thness = et.SubElement(reference_layer, "thickness")
        thness.text = str(self.ref_lay_thickness)
        xmlstr = et.tostring(reference_layer).decode('utf-8')
        element_outdir_name = os.path.join(self.data_dir, "-".join(elements_symbols))
        os.makedirs(element_outdir_name, exist_ok = True)
        out_file_name += f'_{self.ref_lay_thickness}_hydro{self.hydrocerussite_thickness}_live-time{self.live_time}_d-samp-det{self.d_sample_detector}_d-samp-src{self.d_sample_source}_slit{self.slit_size_x}'
        out_file_name += f'_photons{self.n_photons_line}'
        if not self.ofname_is_set:
            self.out_file_name = out_file_name
        if not self.out_file_dirname:
            output_file = self._unique_file(element_outdir_name, self.out_file_name, "xmso")
        else:
            output_file = self._unique_file(self.out_file_dirname, self.out_file_name, "xmso")
        input_template_str = ""
        with open(self.template) as tmplt:
            for line in tmplt:
                line = line.replace('\n', '')
                if "@outputfile@" in line:
                    line = line.replace("@outputfile@", output_file)
                elif "@n_photons_interval@" in line:
                    line = line.replace("@n_photons_interval@", str(self.n_photons_interval))
                elif "@n_photons_line@" in line:
                    line = line.replace("@n_photons_line@", str(self.n_photons_line))
                elif "@reference_layer@" in line:
                    line = line.replace("@reference_layer@", xmlstr)
                elif "@dtd_file@" in line:
                    line = line.replace("@dtd_file@", self.dtd_file)
                elif "@timestamp@" in line:
                    line = line.replace("@timestamp@", timestamp)
                elif "@hydrocerussite_thickness@" in line:
                    line = line.replace("@hydrocerussite_thickness@", str(self.hydrocerussite_thickness))
                elif "@slit_size_x@" in line:
                    line = line.replace("@slit_size_x", str(self.slit_size_x))
                elif "@slit_size_y@" in line:
                    line = line.replace("@slit_size_y@", str(self.slit_size_y))
                elif "@live_time@" in line:
                    line = line.replace("@live_time@", str(self.live_time))
                elif "@nchannels@" in line:
                    line = line.replace("@nchannels@", str(self.nchannels))
                elif "@gain@" in line:
                    line = line.replace("@gain@", str(self.gain))
                elif "@d_sample_detector@" in line:
                    line = line.replace("@d_sample_detector@", str(self.d_sample_detector))
                elif "@d_sample_source@" in line:
                    line = line.replace("@d_sample_source@", str(self.d_sample_source))
                input_template_str += line.strip()
        input_template_str = minidom.parseString(input_template_str).toprettyxml(indent = "  ") 
        element_inputsdir_name = os.path.join(self.input_files_dir, "-".join(elements_symbols))
        os.makedirs(element_inputsdir_name, exist_ok = True)
        if not self.ifname_is_set:
            self.input_file_name = self.out_file_name
        if self.input_file_dirname:
            Ifile = self._unique_file(self.input_file_dirname, self.input_file_name, "xmsi")
        else:
            Ifile = self._unique_file(element_inputsdir_name, self.out_file_name, "xmsi")
        with open(Ifile, "w") as inout:
            inout.write(input_template_str)
        #print(f'input file generated:\n{Ifile}')
        return Ifile
