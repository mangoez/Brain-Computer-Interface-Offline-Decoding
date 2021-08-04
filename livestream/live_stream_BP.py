import argparse
import time
import logging
import random
import socket

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds, BrainFlowError
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
 
def EWA(x_prev, x_new, beta, threshold):
    if abs(x_new) < threshold:
        return (x_prev*beta) + (x_new*(1-beta))
    else:
        return x_prev

class Graph:
    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 50
        self.window_size = 8
        self.partial_window_size = 4
        self.n_pads = 12

        self.noise = np.ones(self.partial_window_size * self.sampling_rate)
        self.beta = np.ones(self.partial_window_size * self.sampling_rate)/10
        self.alpha = np.ones(self.partial_window_size * self.sampling_rate)/10

        self.num_points = self.window_size * self.sampling_rate
        self.num_points_partial = self.partial_window_size * self.sampling_rate
        
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsWindow(title='BrainFlow Plot',size=(800, 600))

        self._init_timeseries()

        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtGui.QApplication.instance().exec_()
    
    def bp_window(self, data, low, high, n_pads):
        padded = np.pad(data[-self.num_points_partial:], n_pads) 
        psd = DataFilter.get_psd(padded, self.sampling_rate, WindowFunctions.BLACKMAN_HARRIS.value)
        bp = DataFilter.get_band_power(psd, low, high) 
        return bp

    def _init_timeseries(self):
        self.plots = list()
        self.curves = list()
        # for i in range(len(self.exg_channels)): # How many time series thingos to display
        for i in range(2):
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
        data = self.board_shim.get_current_board_data(self.num_points)

        if data.shape[1] >= 2000:
            for count, channel in enumerate(self.exg_channels):
                # plot timeseries
                DataFilter.perform_bandpass(data[channel], self.sampling_rate, 25, 25, 2,
                                            FilterTypes.BUTTERWORTH.value, 0)
                DataFilter.perform_bandstop(data[channel], self.sampling_rate, 50.0, 4, 3,
                                            FilterTypes.BUTTERWORTH.value, 0)

            front_all_bp = DataFilter.get_avg_band_powers(data[:, -1000:], [0, 1, 2], self.sampling_rate, False)
            motor_all_bp_L = DataFilter.get_avg_band_powers(data[:, -1000:], [3], self.sampling_rate, False)
            motor_all_bp_R = DataFilter.get_avg_band_powers(data[:, -1000:], [4], self.sampling_rate, False)

            # For calculating beta band power from concentrating
            weighted_beta = EWA(self.beta[-1], front_all_bp[0][3], beta=0.95, threshold=0.11)
            self.beta = np.hstack((self.beta[1:self.num_points_partial], weighted_beta))
            self.curves[0].setData(self.beta)

            # # Calculate alpha band power from motor imagery
            # weighted_alpha = EWA(self.alpha[-1], motor_all_bp_R[0][2]/motor_all_bp_L[0][2], beta=0.99, threshold=1.5)
            # self.alpha = np.hstack((self.alpha[1:self.num_points_partial], weighted_alpha))
            # self.curves[1].setData(self.alpha)
            # print("Weighted Alpha: ", self.alpha[-1], "Weighted Beta: ", self.beta[-1])
            # print("Beta: ", front_all_bp[0][2], "Alpha L: ", motor_all_bp_R[0][3]/motor_all_bp_L[0][3])

            if weighted_beta > 0.104:
                self.alpha = np.hstack((self.alpha[1:self.num_points_partial], 1))
                self.curves[1].setData(self.alpha)
                MESSAGE = '1'
                MESSAGE = bytes(MESSAGE, 'utf-8')
                sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
            else:
                self.alpha = np.hstack((self.alpha[1:self.num_points_partial], 0))
                self.curves[1].setData(self.alpha)
                MESSAGE = '0'
                MESSAGE = bytes(MESSAGE, 'utf-8')
                sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
            print(weighted_beta)                

        self.app.processEvents()


def main():
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

    try:
        board_shim = BoardShim(args.board_id, params)
        board_shim.prepare_session()
        board_shim.start_stream(450000, args.streamer_params)
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
    main()
