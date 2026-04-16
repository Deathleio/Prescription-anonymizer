from ultralytics import YOLO

def train_model():
    # Load the base YOLOv8 Nano model
    model = YOLO('yolov8n.pt')

    # Start training
    model.train(
        data='data.yaml',
        epochs=50,       
        imgsz=640,       
        device=0,        
        batch=8,
        workers=0        # Set to 0 to avoid Windows multiprocessing errors
    )

if __name__ == '__main__':
    train_model()