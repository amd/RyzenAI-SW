from ultralytics import YOLO

def export_yolov8m_to_onnx():
    model = YOLO("yolov8m.pt")
    print("Number of classes:", model.model.nc)
    model.export(format="onnx", opset=17)  # Exports to yolov8m.onnx
    print("YOLOv8m exported to yolov8m.onnx")

if __name__ == "__main__":
    export_yolov8m_to_onnx()
