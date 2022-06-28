import cv2
import numpy as np 

train_data = np.load("data/testingset.npy")

print(train_data.shape)

for i in range(train_data.shape[0]):
    cv2.imwrite(f"tmp/{i}.jpg", cv2.cvtColor(train_data[i], cv2.COLOR_RGB2BGR))
