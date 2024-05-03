from ultralytics import YOLO

# Load a model

model = YOLO('yolov8n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='D:\Image_Classification_By_YOLOV8\WeatherDataset', epochs=20, imgsz=64)

