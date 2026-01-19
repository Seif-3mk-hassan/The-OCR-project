from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from ocr_engine import OCREngine
import pypdfium2 as pdfium
import shutil
import os
import uuid
import numpy as np
from PIL import Image


app = FastAPI(title="Intelligent Document Processing API")

# Initialize OCR Engine
# This might take a moment to download models on the first run
ocr_engine = OCREngine()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def read_root():
    # Serve the frontend HTML using absolute path
    html_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api")
def api_info():
    return {"message": "System Operational", "status": "active"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    handwriting: bool = False,
    extract_tables: bool = False,
    extract_forms: bool = False
):
    if file.content_type not in ["image/jpeg", "image/png", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG, PNG, and PDF are supported.")

    file_extension = file.filename.split(".")[-1] if "." in file.filename else "tmp"
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_location = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    extracted_text = ""
    tables_data = []
    form_data = {}
    
    try:
        if file.content_type == "application/pdf":
            # Convert PDF to images using pypdfium2 (No Poppler required)
            try:
                pdf = pdfium.PdfDocument(file_location)
                for i in range(len(pdf)):
                    page = pdf[i]
                    # Render the page to a PIL image (scale=3 for ~216 DPI, better for OCR)
                    pil_image = page.render(scale=3).to_pil()
                    
                    # Save temp image for table extraction if needed
                    temp_img_path = f"{file_location}_page_{i}.jpg"
                    pil_image.save(temp_img_path)
                    
                    # Convert to numpy for EasyOCR
                    image_np = np.array(pil_image)
                    
                    # Extract text (with handwriting mode if requested)
                    if handwriting:
                        text = ocr_engine.extract_handwritten(image_np)
                    else:
                        text = ocr_engine.extract_text(image_np)
                    
                    extracted_text += f"\n--- Page {i+1} ---\n{text}"
                    
                    # Extract tables if requested
                    if extract_tables:
                        page_tables = ocr_engine.extract_tables(temp_img_path)
                        tables_data.extend(page_tables)
                    
                    # Extract forms if requested
                    if extract_forms:
                        text_blocks = ocr_engine.extract_data(image_np)
                        page_forms = ocr_engine.extract_key_value_pairs(text_blocks)
                        form_data.update(page_forms)
                    
                    # Clean up temp image
                    if os.path.exists(temp_img_path):
                        os.remove(temp_img_path)
                        
            except Exception as e:
                return {
                    "filename": file.filename,
                    "error": f"PDF Processing failed: {e}",
                    "details": str(e)
                }
        else:
            # Image
            if handwriting:
                extracted_text = ocr_engine.extract_handwritten(file_location)
            else:
                extracted_text = ocr_engine.extract_text(file_location)
            
            # Extract tables if requested
            if extract_tables:
                tables_data = ocr_engine.extract_tables(file_location)
            
            # Extract forms if requested
            if extract_forms:
                text_blocks = ocr_engine.extract_data(file_location)
                form_data = ocr_engine.extract_key_value_pairs(text_blocks)
                
    except Exception as e:
        return {"error": f"OCR failed: {str(e)}"}

    return {
        "filename": file.filename,
        "file_id": unique_filename,
        "extracted_text": extracted_text,
        "tables": tables_data if extract_tables else None,
        "form_data": form_data if extract_forms else None
    }

@app.post("/extract-text", response_class=Response)
async def extract_text_only(
    file: UploadFile = File(...),
    handwriting: bool = Query(False, description="Use handwriting recognition model")
):
    """
    Extract text from document and return as plain text (not JSON).
    Useful for simple text extraction without metadata.
    """
    if file.content_type not in ["image/jpeg", "image/png", "application/pdf"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG, PNG, and PDF are supported.")

    file_extension = file.filename.split(".")[-1] if "." in file.filename else "tmp"
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_location = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    extracted_text = ""
    
    try:
        if file.content_type == "application/pdf":
            try:
                pdf = pdfium.PdfDocument(file_location)
                for i in range(len(pdf)):
                    page = pdf[i]
                    pil_image = page.render(scale=3).to_pil()
                    image_np = np.array(pil_image)
                    
                    if handwriting:
                        text = ocr_engine.extract_handwritten(image_np)
                    else:
                        text = ocr_engine.extract_text(image_np)
                    
                    extracted_text += f"\n--- Page {i+1} ---\n{text}"
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"PDF Processing failed: {str(e)}")
        else:
            if handwriting:
                extracted_text = ocr_engine.extract_handwritten(file_location)
            else:
                extracted_text = ocr_engine.extract_text(file_location)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")

    # Return plain text response
    return Response(content=extracted_text, media_type="text/plain; charset=utf-8")

# Mount static files at the end (after all routes)
app.mount("/static", StaticFiles(directory="frontend"), name="static")
