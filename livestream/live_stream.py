import argparse
import time
import logging
import random
from datetime import datetime

import socket
import pygame
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations

import joblib
import math
import yasa
import seaborn as sns
from scipy.signal import welch, freqz, butter, filtfilt, savgol_filter
import mne
from mne.decoding import CSP
from sklearn.model_selection import cross_val_predict

import warnings
warnings.filterwarnings('ignore')

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold

def normalise(x):
    # Electrodes by samples
    m = np.mean(x)
    v = np.var(x, axis=1)
    sd = np.array([math.sqrt(i) for i in v])

    x_norm = np.copy(x)
    for i in range(x_norm.shape[0]):
        x_norm[i, :] = (x[i, :] - m)*(1/sd[i])
    
    return x_norm

def import_data():
    FC1_ind = 0
    FC2_ind = 1
    C3_ind = 2
    Cz_ind = 3
    C4_ind = 4
    CP1_ind = 5
    CP2_ind = 6
    Pz_ind = 7

    data_R = pd.read_csv('data/data_R.csv', index_col=0)
    data_R = data_R.to_numpy()
    print(data_R.shape)
    data_L = pd.read_csv('data/data_L.csv', index_col=0)
    data_L = data_L.to_numpy()
    print(data_L.shape)
    data_Z = pd.read_csv('data/data_Z.csv', index_col=0)
    data_Z = data_Z.to_numpy()
    print(data_Z.shape)

    data_R = np.reshape(data_R, (int(data_R.shape[1]/fs), data_R.shape[0], fs))
    data_L = np.reshape(data_L, (int(data_L.shape[1]/fs), data_L.shape[0], fs))

    data_R = np.reshape(data_R, (data_R.shape[1], data_R.shape[0]*data_R.shape[2]))
    data_L = np.reshape(data_L, (data_L.shape[1], data_L.shape[0]*data_L.shape[2]))

    data = np.hstack((data_R, data_L, data_Z))
    print("Import data shape: ", data.shape)

    n_samples = int(data.shape[1]/length)
    print("Number of samples: ", n_samples)

    y = [1]*int((data_R.shape[1]/fs) + (data_L.shape[1]/fs)) + [0]*int(data_Z.shape[1]/250)
    print("Length of y: ", len(y), "Total positives: ", sum(y))
    print(y)

    
    FC1_cz = data[FC1_ind, :] - data[Cz_ind, :]
    FC2_cz = data[FC2_ind, :] - data[Cz_ind, :]
    C3_cz = data[C3_ind, :] - data[Cz_ind, :]
    C4_cz = data[C4_ind, :] - data[Cz_ind, :]
    CP1_cz = data[CP1_ind, :] - data[Cz_ind, :]
    CP2_cz = data[CP2_ind, :] - data[Cz_ind, :]
    Pz_cz = data[Pz_ind, :] - data[Cz_ind, :]

    FC1 = data[FC1_ind, :] 
    FC2 = data[FC2_ind, :] 
    C3 = data[C3_ind, :] 
    Cz = data[Cz_ind, :] 
    C4 = data[C4_ind, :] 
    CP1 = data[CP1_ind, :] 
    CP2 = data[CP2_ind, :] 
    Pz = data[Pz_ind, :] 

    selected_electrodes_ref = np.vstack((FC1_cz, FC2_cz, C3_cz, C4_cz, CP1_cz, CP2_cz, Pz_cz))
    selected_electrodes = np.vstack((FC1, FC2, C3, Cz, C4, CP1, CP2, Pz))

    n_channels_ref = int(selected_electrodes_ref.shape[0])
    n_channels = int(selected_electrodes.shape[0])

    selected_electrodes_ref_reshaped = np.reshape(selected_electrodes_ref, (n_samples, selected_electrodes_ref.shape[0], length))
    for i in range(selected_electrodes_ref_reshaped.shape[0]):
        selected_electrodes_ref_reshaped[i, :, :] = normalise(selected_electrodes_ref_reshaped[i, :, :])

    selected_electrodes_reshaped = np.reshape(selected_electrodes, (n_samples, selected_electrodes.shape[0], length))
    for i in range(selected_electrodes_reshaped.shape[0]):
        selected_electrodes_reshaped[i, :, :] = normalise(selected_electrodes_reshaped[i, :, :])
        
        
    selected_electrodes_ref = np.reshape(selected_electrodes_ref_reshaped, 
                                        (n_channels_ref, selected_electrodes_ref_reshaped.shape[0]*selected_electrodes_ref_reshaped.shape[2]))

    selected_electrodes = np.reshape(selected_electrodes_reshaped, 
                                    (n_channels, selected_electrodes_reshaped.shape[0]*selected_electrodes_reshaped.shape[2]))

    print("selected_electrodes_reshaped: ", selected_electrodes_reshaped.shape)
    print("selected_electrodes_ref: ", selected_electrodes_ref.shape)
    print("selected_electrodes: ", selected_electrodes.shape)

    return selected_electrodes_reshaped, selected_electrodes_ref, n_channels_ref, n_samples, y

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 8
        self.num_points = self.window_size * self.sampling_rate

        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='BrainFlow Plot',size=(800, 600))

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        
        for i in range(len(self.exg_channels)+1):
            p = self.win.addPlot(row=i,col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('TimeSeries Plot')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def update(self):  
        # received_data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
        # print("Received message: ", received_data)  

        data = self.board_shim.get_current_board_data(self.num_points)

        if data[0:7, 0:250].shape == (7, 250):
            for count, channel in enumerate(self.exg_channels):
                # plot timeseries
                DataFilter.perform_lowpass(data[channel], BoardShim.get_sampling_rate(args.board_id), high, 3,
                                            FilterTypes.BUTTERWORTH.value, 0)
                DataFilter.perform_highpass(data[channel], BoardShim.get_sampling_rate(args.board_id), low, 3,
                                            FilterTypes.BUTTERWORTH.value, 0)
                DataFilter.perform_bandstop(data[channel], BoardShim.get_sampling_rate(args.board_id), 50, 2, 8,
                                            FilterTypes.BUTTERWORTH.value, 0)
                self.curves[count+1].setData(data[channel][-1001:].tolist())
        
            window = data[0:8, -250:] # input window
            window = window - window[3, :] # Cz reference

            x = np.vstack((window[0:3, :], window[4:, :]))
            x = create_data(x, fs, 1, low, high, n_freqs, zeros, length) # convert to PSD
            x = np.reshape(x, (1, n_channels_ref, n_freqs))
            x_csp = csp2.transform(x)

            window = np.reshape(window, (1, window.shape[0], window.shape[1]))
            x_raw_csp = csp1.transform(window)

            inference = np.hstack((x_csp, x_raw_csp))

            current_time = datetime.now()
            current_time = current_time.strftime("%M:%S")
            result.append(model.predict(inference)[0])

            MESSAGE = str(model.predict(inference)[0])
            MESSAGE = bytes(MESSAGE, 'utf-8')
            sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
            pygame.time.delay(100)

            self.curves[0].setData(result[-1001:])

        self.app.processEvents()

def plot_spectrum(data, sf, window_sec, band=None, dB=False):
    sns.set(style="white", font_scale=1.2)
    # Compute the PSD
    freqs_welch, psd_welch = welch(data, sf, nperseg=window_sec*sf)
    sharey = False

    # Optional: convert power to decibels (dB = 10 * log10(power))
    if dB:
        psd_welch = 10 * np.log10(psd_welch)
        sharey = True
    
    return freqs_welch, psd_welch

def yucky_floats(freqs, psd):
    freqs = np.array(freqs)
    temp_freqs = []
    temp_psd = []
    
    for i in range(freqs.shape[0]):
        if float(freqs[i]).is_integer():
            temp_freqs.append(int(freqs[i]))
            temp_psd.append(int(psd[i]))
            
    return np.array(temp_freqs), np.array(temp_psd)

def n_freqs_calc(x, zeros, low, high):
    x_1 = x[0:length]
    x_1_pad = np.pad(x_1, [(zeros, zeros)], mode='constant')

    freqs, psd = plot_spectrum(x_1_pad, fs, (32), [1, 50], dB=True)
    # freqs, psd = yucky_floats(freqs, psd)

    low_ind = np.where(freqs == low)[0][0]
    high_ind = np.where(freqs == high)[0][0]

    psd_sliced = psd[low_ind: high_ind]
    freqs_sliced = freqs[low_ind: high_ind]
    n_freqs = len(psd_sliced)

    return n_freqs, freqs_sliced

def create_data(data, fs, n_samples, low, high, n_freqs, zeros, length):
    channels = data.shape[0]
    x = np.zeros((n_samples, channels*n_freqs))
    temp = np.zeros((channels, n_freqs))
    
    sample_start = 0
    sample_end = length
    
    for i in range(n_samples):
        all_channel_trial = normalise(data[:, sample_start:sample_end])
        
        for j in range(all_channel_trial.shape[0]):
            trial_pad = np.pad(all_channel_trial[j, :], [(zeros, zeros)], mode='constant')
            freqs, psd = plot_spectrum(trial_pad, fs, (32), [low, high], dB=True)

            # freqs, psd = yucky_floats(freqs, psd)
            low_ind = np.where(freqs == low)[0][0]
            high_ind = np.where(freqs == high)[0][0]

            psd_sliced = psd[low_ind: high_ind]
            freqs_sliced = freqs[low_ind: high_ind]

            temp[j, :] = psd_sliced

        sample_start += length
        sample_end += length

        x[i, :] = np.ravel(temp)
        
    return x

def SSC_row(Input):
    # Variable Initialization
    prev = Input[0]/abs(Input[0])
    ans = 0

    # Using Iteration
    for elem in Input:
        elem = int(math.ceil(abs(elem))*(elem/abs(elem)))
        
        if elem == 0:
            sign = -1
        else:
            sign = elem / abs(elem)
        
        if sign == -prev:
            ans = ans + 1
            prev = sign
    return ans

def SSC(Input):
    ans = np.zeros((1, Input.shape[0]))
    
    for r in range(Input.shape[0]):
        row = Input[r, :]
        ans[:,r] = SSC_row(row)
    return np.array(ans)
        
def CSP_data(selected_channels, selected_electrodes_reshaped, selected_electrodes_ref, n_patterns, zeros, n_freqs, y):
    x = create_data(selected_electrodes_ref, fs, n_samples, low, high, n_freqs, zeros, length) 
    print(x.shape)
    x_reshaped = np.reshape(x, (n_samples, n_channels_ref, n_freqs))
    csp1 = CSP(n_components=n_patterns, reg=None, log=True, norm_trace=False)
    csp2 = CSP(n_components=n_patterns, reg=None, log=True, norm_trace=False)

    # Traditional CSP of the raw data
    x_raw_csp = csp1.fit_transform(selected_electrodes_reshaped, y)

    info = mne.create_info(selected_channels, sfreq=fs, ch_types='eeg')
    info['description'] = 'My motor imagery dataset!'
    info.set_montage('standard_1020')

    epochs = mne.EpochsArray(selected_electrodes_reshaped, info)

    # # Apply band-pass filter to the raw data
    # epochs.filter(low, high, fir_design='firwin', skip_by_annotation='edge')

    csp1.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)

    # CSP of the PSD data
    x_csp = csp2.fit_transform(x_reshaped, y)
    print("x_csp shape: ", x_csp.shape)

    return np.hstack((x_csp, x_raw_csp)), x, csp1, csp2




def initialise_livestream():
    BoardShim.enable_dev_board_logger()
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=0)
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='')
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--streamer-params', type=str, help='streamer params', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=BoardIds.SYNTHETIC_BOARD)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    args = parser.parse_args()

    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file

    return params, args

def run_livestream(params, args):
    try:
        board_shim = BoardShim(args.board_id, params)
        board_shim.prepare_session()
        board_shim.start_stream(4500000, args.streamer_params)
        g = Graph(board_shim)
    except BaseException as e:
        logging.warning('Exception', exc_info=True)
    finally:
        logging.info('End')
        if board_shim.is_prepared():
            logging.info('Releasing session')
            board_shim.release_session()


if __name__ == '__main__':
    # Communication with UDP 
    UDP_IP = "127.0.0.1"
    UDP_PORT = 1234 
    sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
    
    fs = 250
    length = int(fs)
    low = 5
    high = 30
    zeros = 375
    n_patterns = 4
    model = ExtraTreesClassifier(n_estimators=500)
    result = []

    # Import all the electrodes 
    electrodes_all = ['FC1', 'FC2', 'C3', 'Cz', 'C4', 'CP1', 'CP2', 'Pz']
    selected_channels_ref = ["FC1", "FC2", "C3", "C4", "CP1", "CP2", "Pz"]
    selected_channels = ["FC1", "FC2", "C3", "Cz", "C4", "CP1", "CP2", "Pz"]

    selected_electrodes_reshaped, selected_electrodes_ref, \
        n_channels_ref, \
            n_samples, \
                y = import_data()

    # Initialise some baord features necessary for filtering
    params, args = initialise_livestream()

    one_channel_one_trial = selected_electrodes_ref[0, 0:length]
    n_freqs, freqs_sliced = n_freqs_calc(one_channel_one_trial, zeros, low, high)

    x_train, x, csp1, csp2 = CSP_data(selected_channels, selected_electrodes_reshaped, selected_electrodes_ref, n_patterns, zeros, n_freqs, y)
    # for i in range(selected_electrodes_ref.shape[0]):
    #     x_L = np.mean(x[0:int(n_samples/3), i*n_freqs:(i*n_freqs)+n_freqs], axis=0)
    #     x_R = np.mean(x[int(n_samples/3):2*int(n_samples/3), i*n_freqs:(i*n_freqs)+n_freqs], axis=0)
    #     x_Z = np.mean(x[2*int(n_samples/3):, i*n_freqs:(i*n_freqs)+n_freqs], axis=0)

    #     plt.figure()
    #     plt.plot(freqs_sliced, x_R)
    #     plt.plot(freqs_sliced, x_L)
    #     plt.plot(freqs_sliced, x_Z)
    #     plt.legend(['R', 'L', 'Z'])
    #     plt.show()

    model.fit(x_train, y)
    # trained_model = joblib.load("data/finalized_model.sav")

    run_livestream(params, args)
