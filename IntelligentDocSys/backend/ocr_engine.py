import easyocr
import logging
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from img2table.document import Image as TableImage
from img2table.ocr import EasyOCR as TableOCR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCREngine:
    def __init__(self, use_gpu=True):
        """
        Initialize the Hybrid OCR engine.
        Uses EasyOCR for Text Detection.
        Uses Microsoft TrOCR for Text Recognition.
        """
        self.use_gpu = use_gpu
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing OCR Engine on {self.device}...")

        # 1. Initialize EasyOCR (Detector only)
        # We still initialize the full reader, but we'll preferentially use detection from it
        self.detector = easyocr.Reader(['en'], gpu=use_gpu)
        logger.info("EasyOCR initialized.")

        # Initialize Table Extraction OCR
        try:
           self.table_ocr = TableOCR(lang='en') # Uses EasyOCR backend for tables
        except Exception as e:
           logger.warning(f"Failed to init Table OCR: {e}")


        # 2. Initialize TrOCR
        try:
            logger.info("Loading TrOCR model (microsoft/trocr-base-printed)...")
            self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
            self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed').to(self.device)
            logger.info("TrOCR model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load TrOCR: {e}")
            raise e

    def load_handwritten_model(self):
        if hasattr(self, 'handwritten_model') and self.handwritten_model is not None:
            return

        try:
            logger.info("Loading TrOCR Handwritten model...")
            self.handwritten_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
            self.handwritten_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten').to(self.device)
            logger.info("TrOCR Handwritten model loaded.")
        except Exception as e:
            logger.error(f"Failed to load Handwritten model: {e}")
            raise e

    def extract_handwritten(self, image_input) -> str:
        """
        Specialized extraction for handwriting using TrOCR Handwritten model.
        """
        self.load_handwritten_model()
        
        # Reuse extract_data logic but swap the recognition model
        # For simplicity, we can refactor extract_data or just duplicate logic slightly for clarity
        # Let's adapt extract_data to accept a 'mode'
        return self.extract_text(image_input, mode="handwritten")

    def extract_text(self, image_input, mode="printed") -> str:
        """
        Extract text from an image.
        Mode: 'printed' (uses EasyOCR) or 'handwritten' (uses TrOCR Handwritten)
        """
        try:
            results = self.extract_data(image_input, mode=mode)
            # Concatenate all recognized text
            full_text = "\n".join([item[1] for item in results])
            return full_text
        except Exception as e:
            logger.error(f"Error during text extraction: {e}")
            return ""

    def extract_data(self, image_input, mode="printed"):
        """
        Extract bounding boxes and recognize text.
        """
        try:
            # 1. Load Image
            if isinstance(image_input, str):
                image = Image.open(image_input).convert("RGB")
                image_np = np.array(image)
            else:
                image_np = image_input
                image = Image.fromarray(image_input).convert("RGB")

            # 2. Detect Text Boxes using EasyOCR
            logger.info("Detecting text with EasyOCR...")
            detection_results = self.detector.readtext(image_np)
            
            hybrid_results = []
            
            for (bbox, easy_text, easy_conf) in detection_results:
                # bbox format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                
                # 3. Crop Image based on bbox
                # Get min/max coordinates with Padding
                padding = 10 
                np_box = np.array(bbox)
                x_min = max(0, int(np.min(np_box[:, 0])) - padding)
                x_max = min(image.width, int(np.max(np_box[:, 0])) + padding)
                y_min = max(0, int(np.min(np_box[:, 1])) - padding)
                y_max = min(image.height, int(np.max(np_box[:, 1])) + padding)
                
                if x_max - x_min < 5 or y_max - y_min < 5:
                    continue # Skip tiny boxes
                    
                cropped_image = image.crop((x_min, y_min, x_max, y_max))

                final_text = ""
                
                if mode == "handwritten":
                    # Use TrOCR Handwritten
                     # Upscale by 3x to give TrOCR more pixels
                    from PIL import ImageEnhance
                    cropped_image = cropped_image.resize((cropped_image.width * 3, cropped_image.height * 3), Image.BICUBIC)
                    enhancer = ImageEnhance.Contrast(cropped_image)
                    cropped_image = enhancer.enhance(2.0) 

                    pixel_values = self.handwritten_processor(images=cropped_image, return_tensors="pt").pixel_values.to(self.device)
                    with torch.no_grad():
                        generated_ids = self.handwritten_model.generate(pixel_values)
                    final_text = self.handwritten_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    logger.info(f"Handwritten: '{final_text}'")

                else:
                    # Default: Printed (EasyOCR)
                    final_text = easy_text
                    logger.info(f"Detected (Printed): '{final_text}'")
                
                hybrid_results.append((bbox, final_text, easy_conf if mode == "printed" else 1.0))

            return hybrid_results

        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            return []

    def save_debug_image(self, image, results, output_path="debug_visualization.jpg"):
        """
        Draws bounding boxes on the image and saves it for debugging.
        """
        try:
            from PIL import ImageDraw
            debug_image = image.copy()
            draw = ImageDraw.Draw(debug_image)
            
            for (bbox, text, _) in results:
                # bbox is usually a list of lists: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                # Flatten it or use polygon
                xy_coords = [tuple(point) for point in bbox]
                draw.polygon(xy_coords, outline="red", width=3)
            
            debug_image.save(output_path)
            logger.info(f"Debug image saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save debug image: {e}")

    def extract_tables(self, image_path: str) -> list:
        """
        Extracts tables from an image path using img2table.
        Returns a list of HTML representations of tables or structured data.
        """
        try:
            doc = TableImage(src=image_path)
            
            # Extract tables
            extracted_tables = doc.extract_tables(ocr=self.table_ocr, implicit_rows=False, borderless_tables=False, min_confidence=50)
            
            tables_data = []
            for table in extracted_tables:
                # Convert to HTML for easy rendering/JSON inclusion
                html = table.html
                tables_data.append(html)
                
            return tables_data
        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return []
    def extract_key_value_pairs(self, text_blocks):
        """
        Heuristic-based Key-Value extraction.
        Assumes "Key: Value" or "Key Value" structure with horizontal alignment.
        Args:
            text_blocks: list of (bbox, text, conf) tuples from extract_data/extract_text
        """
        # Simple heuristic: Look for lines ending in ':' or standard keywords
        kv_pairs = {}
        
        # Sort blocks by Y (top to bottom), then X (left to right)
        # Note: bbox is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        # We use y1 of top-left corner for sorting rows
        sorted_blocks = sorted(text_blocks, key=lambda b: (b[0][0][1], b[0][0][0]))
        
        used_indices = set()
        
        for i, block in enumerate(sorted_blocks):
            if i in used_indices:
                continue
                
            text = block[1].strip()
            bbox = block[0]
            
            # Check if this looks like a key
            is_key = text.endswith(":") or text.isupper() # Naive heuristic
            
            if is_key:
                key = text.strip(":")
                
                # Look for a value to the right of this key
                # We define "right" as: same Y-range (roughly), X > key_X
                key_y_min = bbox[0][1]
                key_y_max = bbox[2][1]
                key_x_max = bbox[1][0]
                
                best_value = None
                best_idx = -1
                
                for j, potential_val in enumerate(sorted_blocks):
                    if i == j or j in used_indices:
                        continue
                        
                    val_bbox = potential_val[0]
                    val_y_min = val_bbox[0][1]
                    val_x_min = val_bbox[0][0]
                    
                    # Check vertical overlap (same line)
                    # Overlap if max(y_min_1, y_min_2) < min(y_max_1, y_max_2)
                    # Simplified: Centers are close in Y
                    val_y_center = (val_bbox[0][1] + val_bbox[2][1]) / 2
                    key_y_center = (key_y_min + key_y_max) / 2
                    
                    if abs(val_y_center - key_y_center) < 20: # 20px tolerance for same line
                        if val_x_min > key_x_max: # It's to the right
                            if best_value is None or val_x_min < best_value[0][0][0]: # Closest to the right
                                best_value = potential_val
                                best_idx = j
                
                if best_value:
                    kv_pairs[key] = best_value[1]
                    used_indices.add(i)
                    used_indices.add(best_idx)
                    logger.info(f"Found KV Pair: {key} -> {best_value[1]}")
        
        return kv_pairs
