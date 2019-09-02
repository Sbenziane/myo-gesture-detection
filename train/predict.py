from models import TwoLayerNet

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import collections
import myo
import threading


model_path = 'models/model_fist_fingers_spread.pt'


LABELLEN = 5
THRESHOLD = 0.5

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 2048, 100, LABELLEN
epochs = 20
batch_size = 16
queue_size = 512


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


def main(model):
    global queue_size

    # Initialize Myo and create a Hub and our listener.
    myo.init(sdk_path='../create_dataset/myo_sdk/')
    hub = myo.Hub()
    listener = Listener(queue_size)

    def get_data():
        global count

        emgs = np.array([x[1] for x in listener.get_emg_data()]).T
        # print('emgs')
        # print(emgs)
        if emgs.shape == (8, queue_size):
            f = emgs
            F = np.fft.fft(f)
            Amp = np.abs(F)
            first_Amp = Amp[:, 0:int(queue_size/2)]

            # size: 8*queue_size/2
            flat_Amp = np.reshape(first_Amp, (1, int(8*queue_size/2)))[0]

            # save_data = np.hstack((label, flat_Amp))

            # size: (1,len(label) + 8*queue_size/2)
            # save_data = np.array([save_data])
            input_data = torch.from_numpy(flat_Amp).float()
            # model predict
            pred = model(input_data)
            # print(pred)

            pred_numpy = pred.detach().numpy()

            finger_data = np.empty((0, LABELLEN))
            # print(finger_data)
            for i, d in enumerate(pred_numpy):
                if d > THRESHOLD:
                    finger_data = np.append(finger_data, 1)
                else:
                    finger_data = np.append(finger_data, 0)

            print(finger_data)

        else:
            print("buffering")
            # print(emgs)

    try:
        threading.Thread(target=lambda: hub.run_forever(
            listener.on_event)).start()

        while True:
            get_data()

        print('saving data...')
        print(all_data.shape)
        write_csv(all_data, 'database1.csv')
        print('finish')
    finally:
        hub.stop()


if __name__ == "__main__":
    print('start!')
    model = TwoLayerNet(D_in, H, D_out)
    model.load_state_dict(torch.load(model_path))
    main(model)
