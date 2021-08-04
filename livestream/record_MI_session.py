import time
import pygame
from random import seed
import random
from random import randint
import numpy as np
from argparse import ArgumentParser
from datetime import datetime
from scipy.signal import welch, freqz, butter, filtfilt
import matplotlib.pyplot as plt
from datetime import datetime
from pygame.locals import *
import pandas as pd

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations


# set up stuff
y = []
left = pygame.image.load('pictures/left.png')
right = pygame.image.load('pictures/right.png')
colour = (102, 0, 204)

# Pygame stuff
def redraw_bg(surface):
    surface.fill((0, 0, 0))
    pygame.draw.circle(surface, (160, 160, 160), (840,525), 50)

def draw_black(surface):
    surface.fill((0,0,0))

def draw_stimuli(surface, direction):
    draw_black(surface)
    x = 525
    y = 250
    if (direction == 1):
        surface.blit(right, (x,y))
    #     pygame.draw.polygon(surface, (160, 160, 160), ((400, 500), (400, 550), (1300, 550), (1300, 625), (1400, 530), (1300, 430), (1300, 500)))
    #     y.append(1)
    elif (direction == -1):
        surface.blit(left, (x,y))
    #     pygame.draw.polygon(surface, (160, 160, 160), ((400, 500), (400, 430), (300, 525), (399, 620), (400, 550), (1300, 550), (1300, 500)))
    #     y.append(-1)
    else:
        None

# Brainflow stuff
def initialise_brainflow():
    BoardShim.enable_dev_board_logger()

    parser = ArgumentParser()
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
                        required=True)
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
    params.board_id = args.board_id

    return params, args

def start_recording(total_sec, max_samples, params, args):
    board = BoardShim(args.board_id, params)
    board.prepare_session()

    # board.start_stream () # use this for default options
    board.start_stream(max_samples, args.streamer_params)
    time.sleep(total_sec)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    data = board.get_board_data()  # get all data and remove it from internal buffer
    timestamp_channel = board.get_timestamp_channel(board_id=0)

    board.stop_stream()
    board.release_session()


    # demo how to convert it to pandas DF and plot data
    eeg_channels = BoardShim.get_eeg_channels(args.board_id)
    print(eeg_channels)
    df = pd.DataFrame(np.transpose(data))
    plt.figure()
    df[eeg_channels].plot(subplots=True)
    plt.savefig('data/before_processing.png')

    # for demo apply different filters to different channels, in production choose one
    for count, channel in enumerate(eeg_channels):
        DataFilter.perform_lowpass(data[channel], BoardShim.get_sampling_rate(args.board_id), 30, 3,
                                        FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_highpass(data[channel], BoardShim.get_sampling_rate(args.board_id), 5, 3,
                                        FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(data[channel], BoardShim.get_sampling_rate(args.board_id), 50, 2, 8,
                                        FilterTypes.BUTTERWORTH.value, 0)


    df = pd.DataFrame(np.transpose(data[:, 1000:])) # Usable after 1000 given order of filters are 3
    plt.figure()
    df[eeg_channels].plot(subplots=True)
    plt.savefig('data/after_processing.png')

    return data, timestamp_channel


# Action!
def save_data(new_data, direction):
    date = datetime.today().strftime('%Y_%m_%d_%H_%M_%S')

    if (direction == 2):
        pd.DataFrame(new_data).to_csv("data/R_" + str(date) + ".csv")
    elif (direction == 1):
        pd.DataFrame(new_data).to_csv("data/L_" + str(date) + ".csv")
    else:
        pd.DataFrame(new_data).to_csv("data/Z_" + str(date) + ".csv")

def countdown(surface, sec):
    x = 750
    y = 300
    surface.fill((0, 0, 0))
    myfont = pygame.font.SysFont('Comic Sans MS', 300)

    for i in range(sec, 0, -1):
        surface.fill((0, 0, 0))
        text = myfont.render(str(i), False, (160, 160, 160))
        surface.blit(text,(x, y))
        pygame.display.update()
        pygame.time.delay(1000)

def accept_direction(surface, event):
    myfont = pygame.font.SysFont('Comic Sans MS', 100)

    if event == 0:
        text = myfont.render('Rest now, ready?', False, colour)
    elif event == 1:
        text = myfont.render('Imagine left hand, ready?', False, colour)
    else:
        text = myfont.render('Imagine right hand, ready?', False, colour)

    surface.blit(text,(350, 400))
    pygame.display.update()

    pygame.event.clear()
    key_not_pressed = True
    answer = 0
    while key_not_pressed:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                key_not_pressed = False

    return None

def record_now():
    # Initialise shit
    sec_per_trial = []
    shapes_per_trial = []
    params, args = initialise_brainflow()

    n_trials = 3
    directions = [0]*n_trials*2 + [1]*n_trials + [2]*n_trials
    random.shuffle(directions)

    # Initialise Pygame
    pygame.init()
    # win = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    win = pygame.display.set_mode((1620, 800))

    pygame.display.set_caption("MI stimulation")

    fs = 250
    sec = 12

    for direct in directions:
        draw_black(win)
        accept_direction(win, direct) # Receive input from user
        print("DIRECTION: ", direct)
        pygame.display.update()

        # redraw_bg(win)
        # pygame.display.update()
        countdown(win, 2)

        draw_stimuli(win, direct)
        pygame.display.update()

        data, timestamp_channel = start_recording(sec, sec*1100, params, args)
        data = np.array(data)
        single_trial_timestamps = data[timestamp_channel, -((sec-4)*fs)-1:-1]
        single_trial_data = data[1:9, -((sec-4)*fs)-1:-1]

        start = single_trial_timestamps[0]
        end = single_trial_timestamps[-1]

        draw_black(win)
        pygame.display.update()

        sec_per_trial.append(end-start)
        shapes_per_trial.append(single_trial_data.shape)

        print(sec_per_trial, shapes_per_trial)
        save_data(single_trial_data, direct)


if __name__ == "__main__":
    # Start recording stuff
    record_now()
    pygame.quit()
