from ultralytics import YOLO

import numpy as np


model = YOLO("D:\\Image_Classification_By_YOLOV8\\runs\classify\\train5\\weights\\last.pt")  # load a custom model

results = model("D:\Image_Classification_By_YOLOV8\WeatherDataset\\train\cloudy\cloudy136.jpg")  # predict on an image

results1 = model("D:\Image_Classification_By_YOLOV8\WeatherDataset\\train\sunrise\sunrise269.jpg")  # predict on an image


names_dict = results[0].names
names_dict1 = results1[0].names


probs = results[0].probs.data.tolist()
probs1 = results1[0].probs.data.tolist()

print(names_dict)
print(names_dict1)

print(probs)
print(probs1)

print(names_dict[np.argmax(probs)])
print(names_dict1[np.argmax(probs1)])
