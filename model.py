from ultralytics import YOLO

model = YOLO("yolov8m.pt")
results = model.predict("cat_dog.jpg")
result = results[0]

for box in result.boxes:
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    class_id = result.names[box.cls[0].item()]
    conf = round(box.conf[0].item(), 2)

    print("Object type:", class_id)
    print("Coordinates:", cords)
    print("Probability:", conf)

from PIL import Image
Image.fromarray(result.plot()[:,:,::-1]).show()