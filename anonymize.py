import os
import cv2
import numpy as np
from ultralytics import YOLO
import fitz  # PyMuPDF for reading PDFs
from PIL import Image # Pillow for robust image loading

def apply_anonymization(img, model):
    """
    Helper function to run YOLO and draw black boxes on a single image array.
    """
    # 1. Ask the AI to find the PII blocks in the image
    results = model(img, verbose=False) 
    
    # 2. Extract the coordinates for every block it finds
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get the top-left (x1, y1) and bottom-right (x2, y2) coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 3. Draw a solid black rectangle over the detected PII region
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
            
    return img

def anonymize_prescriptions(input_dir, output_dir, model_path):
    # Create the output folder if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load your newly trained custom model
    print(f"Loading custom model from: {model_path}")
    model = YOLO(model_path)
    
    # Added .pdf to valid extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.pdf')
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(valid_extensions):
            continue
            
        img_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # ==========================================
        # HANDLE PDF FILES
        # ==========================================
        if filename.lower().endswith('.pdf'):
            try:
                pdf_document = fitz.open(img_path)
                anonymized_pages = []
                
                # Loop through every page in the PDF
                for page_num in range(len(pdf_document)):
                    page = pdf_document.load_page(page_num)
                    # Render page to an image
                    pix = page.get_pixmap(dpi=300) 
                    
                    # Convert the PyMuPDF pixmap to a NumPy array for OpenCV
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                    
                    # Convert to standard OpenCV BGR format
                    if pix.n == 4: # If it has an alpha/transparency channel
                        cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                    else:
                        cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        
                    # Send to our YOLO helper function
                    anon_cv_img = apply_anonymization(cv_img, model)
                    
                    # Convert back to a PIL Image to save as PDF
                    anon_pil_img = Image.fromarray(cv2.cvtColor(anon_cv_img, cv2.COLOR_BGR2RGB))
                    anonymized_pages.append(anon_pil_img)
                    
                # Save all the anonymized pages back into a single PDF
                if anonymized_pages:
                    anonymized_pages[0].save(
                        output_path, 
                        save_all=True, 
                        append_images=anonymized_pages[1:], 
                        resolution=300.0
                    )
                print(f"Successfully anonymized PDF: {filename}")
                
            except Exception as e:
                print(f"Failed to process PDF {filename}: {str(e)}")

        # ==========================================
        # HANDLE STANDARD IMAGE FILES (.png, .jpg)
        # ==========================================
        else:
            try:
                # FIXED: Use Pillow instead of cv2.imread to safely handle PNG transparency/palettes
                pil_img = Image.open(img_path).convert('RGB')
                
                # Convert Pillow RGB format into OpenCV BGR array
                img_array = np.array(pil_img)
                cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                # Send to our YOLO helper function
                anon_cv_img = apply_anonymization(cv_img, model)
                
                # Save the safely anonymized image to the output folder
                cv2.imwrite(output_path, anon_cv_img)
                print(f"Successfully anonymized Image: {filename}")
                
            except Exception as e:
                print(f"Failed to process image {filename}: {str(e)}")

if __name__ == '__main__':
    # Folders
    INPUT_FOLDER = 'raw_prescriptions' 
    OUTPUT_FOLDER = 'safe_prescriptions_for_ocr'
    
    # Pointing exactly to your successful training run
    MODEL_PATH = r'runs\detect\train8\weights\best.pt' 
    
    print("Starting batch anonymization process...")
    anonymize_prescriptions(INPUT_FOLDER, OUTPUT_FOLDER, MODEL_PATH)
    print(f"Done! Check the '{OUTPUT_FOLDER}' folder.")