# ************************************************************************************************************* #
#                                                                                                               #
# OBJECTIVE : DATA MINING PROJECT                                               #####      ###    ###    #      #
# AUTHORS :  VICTOR DALET , LUCA GUILLIERE, TRIBOUT NOLHAN                      #         #      #       #      #
# CREATED : 15 11 2023                                                          ####      #      #  ##   #      #
# UPDATE  : 17 11 2023                                                          #         #      #   #   #      #
#                                                                               ####    ###      #####   #.fr   #
# ************************************************************************************************************* #

import matplotlib.pyplot as plt
import pandas as pd
from binaryNeural import BinaryNeural
import tensorflow as tf
import csv as csv_write

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    tf.config.set_visible_devices(gpus[0], 'GPU')
    csv = pd.read_csv("train.csv")
    data: list[list] = csv.values.tolist()
    label: list[str] = list(csv)
    inputs: list[list[int]] = []
    outputs: list[int] = []
    for d in data:
        inputs.append(d[1:-1])
        outputs.append(d[-1])

    print(len(inputs), len(outputs))
    for i in range(len(inputs[0])):
        array_temp = []
        for j in range(len(inputs)):
            array_temp.append(inputs[j][i])
        plt.scatter(x=outputs, y=array_temp)
        plt.title(label[i])
        plt.show()

    dwayne: BinaryNeural = BinaryNeural(inputs[:10000], outputs, 10000, .01)
    dwayne.train()

    csv = pd.read_csv("test.csv")
    data: list[list] = csv.values.tolist()
    inputs: list[list[int]] = []
    for d in data:
        inputs.append(d[1:])
    for i in range(len(inputs)):
        f = open('sample_submission.csv', 'a', newline='')
        writer = csv_write.writer(f)
        writer.writerow([int(data[i][0]), int(dwayne.predict(inputs[i]))])
        f.close()
