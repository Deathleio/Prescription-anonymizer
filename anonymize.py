import os
import cv2
from ultralytics import YOLO

def anonymize_prescriptions(input_dir, output_dir, model_path):
    # Create the output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load your newly trained custom model
    print(f"Loading custom model from: {model_path}")
    model = YOLO(model_path)
    
    valid_extensions = ('.jpg', '.jpeg', '.png')
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(valid_extensions):
            continue
            
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Could not read {filename}. Skipping.")
            continue
            
        # 1. Ask the AI to find the PII blocks in the image
        results = model(img, verbose=False) 
        
        # 2. Extract the coordinates for every block it finds
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get the top-left (x1, y1) and bottom-right (x2, y2) coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 3. Draw a solid black rectangle over the detected PII region
                # (0, 0, 0) is black. -1 tells OpenCV to fill the rectangle completely.
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

        # 4. Save the safely anonymized image to the output folder
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, img)
        print(f"Successfully anonymized: {filename}")

if __name__ == '__main__':
    # Folders
    INPUT_FOLDER = 'raw_prescriptions' 
    OUTPUT_FOLDER = 'safe_prescriptions_for_ocr'
    
    # Pointing exactly to your successful training run!
    MODEL_PATH = r'runs\detect\train2\weights\best.pt' 
    
    print("Starting batch anonymization process...")
    anonymize_prescriptions(INPUT_FOLDER, OUTPUT_FOLDER, MODEL_PATH)
    print(f"Done! Check the '{OUTPUT_FOLDER}' folder.")