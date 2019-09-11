from __future__ import print_function
from matplotlib import pyplot as plt
from util import write_csv, data_append, normalize, sigmoid
import collections
import myo
import numpy as np
import threading
import time

queue_size = 512
DATANUM_TOTAL = 7000
DATANUM_EACH = 100

LABELLEN = 5
SENSORNUM = 8
all_data = []
count = 0
flg_get_data = True

LABELS = [[0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0],
          [0, 1, 0, 0, 0],
          [0, 0, 1, 0, 0],
          [0, 0, 0, 1, 0],
          [0, 0, 0, 0, 1],
          [1, 1, 1, 1, 1]]

LEBELS_NUM = len(LABELS)  # 7


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


def main():
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
            if count > DATANUM_TOTAL-2:
                flg_get_data = False

            label_index = int(count / DATANUM_EACH) % LEBELS_NUM
            label = LABELS[label_index]
            label = np.array(label).astype('int32')
            print(
                f'{count+1}/{DATANUM_TOTAL} label_index: {label_index} {LABELS[label_index]}')
            count += 1

            # print(type(emgs))
            # print(emgs.shape)
            # print(emgs[0])
            f = emgs
            m = np.mean(f, axis=1)
            v = np.var(f, axis=1)
            # m_norm = normalize(m)
            # print(m_norm)
            # print(m)
            # print(normalize(v))
            # print(v)
            # print(sigmoid(v - np.mean(v)))
            # v_chg = sigmoid(v - np.mean(v))
            m = np.mean(np.abs(f), axis=1)
            # print(np.mean(np.abs(f), axis=1))
            F = np.fft.fft(f)
            Amp = np.abs(F)
            first_Amp = Amp[:, 0:int(queue_size/2)]

            # size: 8*queue_size/2
            flat_Amp = np.reshape(first_Amp, (1, int(8*queue_size/2)))[0]
            flat_Amp_norm = normalize(flat_Amp)
            # print(flat_Amp_norm)
            # size: len(label) + 8*queue_size/2
            # save_data = np.hstack((label, flat_Amp))
            save_data = np.hstack((label, m, flat_Amp_norm))

            save_data = list(save_data)
            # print('save_data', save_data)
            # print(save_data)
            all_data = data_append(all_data, save_data)

        else:
            print("buffering")
            # print(emgs)

    try:
        threading.Thread(target=lambda: hub.run_forever(
            listener.on_event)).start()

        while flg_get_data:
            get_data()
            time.sleep(0.1)

        print('saving data...')
        print(np.array(all_data).shape)
        write_csv(all_data, SAVE_DATA_PATH)
        print('finish', SAVE_DATA_PATH)
    finally:
        hub.stop()


if __name__ == '__main__':
    print("input finger situation")
    # finger_situation = input()
    # SAVE_DATA_PATH = 'dataset/dataset_0904_2_' + finger_situation + '.csv'
    SAVE_DATA_PATH = 'dataset/var/dataset_0910_1_seq.csv'
    print(SAVE_DATA_PATH)
    # 0→extended
    # 1→not extended
    # finger_situation_ary = list(finger_situation)
    # if len(finger_situation_ary) == 5:
    #     print(finger_situation_ary)
    # else:
    #     print('the length is not 5!!')
    # main(finger_situation_ary)
    main()
