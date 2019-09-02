from websocket import create_connection
import websocket
import json
import pprint
import ast

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ws = create_connection("ws://localhost:6437/v6.json")
result = ws.recv()
print("Received '%s'" % result)

enableMessage = {'enableGestures': True}
json_enableMessage = json.dumps(enableMessage)
# {focused: true}
ws.send(json_enableMessage)
ws.send(json.dumps({'focused': True}))

result = ws.recv()
print("Received '%s'" % result)


print("Sent")
print("Receiving...")

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
# sc, =  ax.scatter(0, 0, 0, c='r', marker='o')

c = 0
while True:
    c += 1

    print('-'*20)
    result = ws.recv()
    result = result.replace('true', 'True').replace(
        'false', 'False').replace('null', 'None')
    # result_dic = ast.literal_eval(result)
    result_dic = eval(result)

    # pprint.pprint(result_dic)

    try:
        extended = []
        cos_theta = []
        position = []

        x = []
        y = []
        z = []
        for i in range(5):
            if result_dic["pointables"][i]["extended"]:
                extended.append(1)
            else:
                extended.append(0)

            stabilizedTipPosition = result_dic["pointables"][i]["stabilizedTipPosition"]
            tipPosition = result_dic["pointables"][i]["tipPosition"]
            dipPosition = result_dic["pointables"][i]["dipPosition"]
            pipPosition = result_dic["pointables"][i]["pipPosition"]
            mcpPosition = result_dic["pointables"][i]["mcpPosition"]
            palmPosition = result_dic["hands"][0]['palmPosition']

            # print(type(x), tipPosition[0])
            x.append(tipPosition[0])
            x.append(dipPosition[0])
            x.append(pipPosition[0])
            x.append(mcpPosition[0])
            x.append(palmPosition[0])

            y.append(tipPosition[1])
            y.append(dipPosition[1])
            y.append(pipPosition[1])
            y.append(mcpPosition[1])
            y.append(palmPosition[1])

            z.append(tipPosition[2])
            z.append(dipPosition[2])
            z.append(pipPosition[2])
            z.append(mcpPosition[2])
            z.append(palmPosition[2])

            # print(palmPosition)
            if i == 0:  # thumb
                v_base = np.array(pipPosition) - np.array(mcpPosition)
                v_tip = np.array(tipPosition) - np.array(dipPosition)
            else:
                v_base = np.array(mcpPosition) - np.array(palmPosition)
                v_tip = np.array(tipPosition) - np.array(mcpPosition)

            # print(v_tip, tipPosition, dipPosition)

            cos_theta_i = (np.dot(v_base, v_tip)) / \
                (np.linalg.norm(v_base) * np.linalg.norm(v_tip))
            cos_theta.append(cos_theta_i)

        # position = np.concatenate([[stabilizedTipPosition],
        #                            [dipPosition], [pipPosition], [mcpPosition]]).T

        # print('stabilizedTipPosition', stabilizedTipPosition)
        # print('dipPosition', dipPosition)
        # print('pipPosition', pipPosition)
        # print('mcpPosition', mcpPosition)

        # pprint.pprint(position)
        color = ['r', 'g', 'b', 'c', 'y', 'r', 'g', 'b', 'c', 'y', 'r', 'g',
                 'b', 'c', 'y', 'r', 'g', 'b', 'c', 'y', 'r', 'g', 'b', 'c', 'y', ]
        # for i, p in enumerat e(position):

        # sc.scatter(position[0], position[1], position[2], c=color, marker='o')
        # print(len(x))
        # ax.scatter(x, y, z, c=color, marker='o')
        ac_theta = 180 - np.rad2deg(np.arccos(cos_theta))
        print('{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            cos_theta[0], cos_theta[1], cos_theta[2], cos_theta[3], cos_theta[4]))
        print('{:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(
            ac_theta[0], ac_theta[1], ac_theta[2], ac_theta[3], ac_theta[4]))

        # print(extended)

        # plt.show()

        # break

        # if c < 5:
        #     plt.draw()
        #     plt.pause(0.01)
        # plt.show()

    except IndexError as e:
        print(e)


ws.close()
