from __future__ import print_function
from matplotlib import pyplot as plt
from util import write_csv
import collections
import myo
import numpy as np
import threading
import time

queue_size = 512
DATANUM = 3000


LABELLEN = 5
SENSORNUM = 8
all_data = np.empty((0, LABELLEN + int(SENSORNUM*queue_size/2)))
count = 0
flg_get_data = True


class Listener(myo.DeviceListener):

    def __init__(self, queue_size=8):
        self.lock = threading.Lock()
        self.emg_data_queue = collections.deque(maxlen=queue_size)

    def on_connected(self, event):
        event.device.stream_emg(True)

    def on_pose(self, event):
        self.pose = event.pose
        # print(self.pose)
        # if self.pose == myo.Pose.double_tap:
        #     event.device.stream_emg(True)
        #     self.emg_enabled = True
        # elif self.pose == myo.Pose.fingers_spread:
        #     event.device.stream_emg(False)
        #     self.emg_enabled = False
        #     self.emg = None

    def on_emg(self, event):
        with self.lock:
            emg = event.emg
            e_time = event.timestamp
            # print((e_time, emg))
            self.emg_data_queue.append((event.timestamp, emg))
            # print('start list')
            # print(list(self.emg_data_queue))
            # print('end list')

    def get_emg_data(self):
        with self.lock:
            return list(self.emg_data_queue)


def main(label):
    global queue_size

    # Initialize Myo and create a Hub and our listener.
    myo.init(sdk_path='./myo_sdk/')
    hub = myo.Hub()
    listener = Listener(queue_size)

    def get_data():
        global count
        global flg_get_data
        global all_data

        emgs = np.array([x[1] for x in listener.get_emg_data()]).T
        # print('emgs')
        # print(emgs)
        if emgs.shape == (8, queue_size):
            if count > DATANUM-2:
                flg_get_data = False

            print(f'{count+1}/{DATANUM}')
            count += 1

            # print(type(emgs))
            # print(emgs.shape)
            # print(emgs[0])
            f = emgs
            F = np.fft.fft(f)
            Amp = np.abs(F)
            first_Amp = Amp[:, 0:int(queue_size/2)]

            # print(Amp)
            # print(first_Amp)

            # size: 8*queue_size/2
            flat_Amp = np.reshape(first_Amp, (1, int(8*queue_size/2)))[0]

            # print(label, flat_Amp)

            # size: len(label) + 8*queue_size/2
            save_data = np.hstack((label, flat_Amp))

            # size: (1,len(label) + 8*queue_size/2)
            save_data = np.array([save_data])
            # print(save_data)

            # print(all_data, save_data)
            all_data = np.append(all_data, save_data, axis=0)
            # print(all_data)

        else:
            print("buffering")
            # print(emgs)

    try:
        threading.Thread(target=lambda: hub.run_forever(
            listener.on_event)).start()

        while flg_get_data:
            get_data()

        print('saving data...')
        print(all_data.shape)
        write_csv(all_data, SAVE_DATA_PATH)
        print('finish', SAVE_DATA_PATH)
    finally:
        hub.stop()


if __name__ == '__main__':
    print("input finger situation")
    finger_situation = input()
    SAVE_DATA_PATH = 'dataset/dataset_0903_3_' + finger_situation + '.csv'
    print(SAVE_DATA_PATH)
    # 0→extended
    # 1→not extended
    finger_situation_ary = list(finger_situation)
    if len(finger_situation_ary) == 5:
        print(finger_situation_ary)
    else:
        print('the length is not 5!!')
    main(finger_situation_ary)
