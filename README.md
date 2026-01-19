# 📄 Intelligent Document Processing System

A powerful web-based document processing system that extracts text from PDF and image files using advanced OCR technology. Features a modern, beautiful web interface and supports printed text, handwritten content, table extraction, and form processing.

![Web Interface](https://img.shields.io/badge/Interface-Modern-purple) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)

## ✨ Features

- **📝 Text Extraction** - Extract text from PDFs, JPG, and PNG files
- **✍️ Handwriting Recognition** - Specialized model for handwritten documents (TrOCR-Handwritten)
- **📊 Table Extraction** - Automatically detect and extract table structures  
- **📋 Form Processing** - Extract key-value pairs from forms using spatial heuristics
- **🎨 Modern Web UI** - Beautiful purple gradient interface for easy document upload
- **🔌 REST API** - Full API access with automatic documentation

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or navigate to the project**
   ```bash
   cd IntelligentDocSys/backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server**
   ```bash
   uvicorn main:app --reload
   ```

4. **Access the application**
   - **Web Interface**: http://127.0.0.1:8000
   - **API Documentation**: http://127.0.0.1:8000/docs

## 💻 Usage

### Web Interface

1. Open http://127.0.0.1:8000 in your browser
2. Click "Choose File" and select a PDF, JPG, or PNG document
3. (Optional) Check boxes for Handwriting Recognition, Table Extraction, or Form Processing
4. Click "Extract Text"
5. View extracted text in the text area
6. Click "Copy to Clipboard" to copy the results

### API Endpoints

#### Extract Text (Plain Text Response)
```bash
# Basic extraction
curl -X POST "http://127.0.0.1:8000/extract-text" -F "file=@document.pdf"

# Handwriting recognition
curl -X POST "http://127.0.0.1:8000/extract-text?handwriting=true" -F "file=@note.jpg"
```

#### Upload (JSON Response with All Features)
```bash
# Extract tables and forms
curl -X POST "http://127.0.0.1:8000/upload?extract_tables=true&extract_forms=true" \
  -F "file=@invoice.pdf"

# Handwriting with all features
curl -X POST "http://127.0.0.1:8000/upload?handwriting=true&extract_tables=true&extract_forms=true" \
  -F "file=@handwritten_form.jpg"
```

#### Response Format
```json
{
  "filename": "document.pdf",
  "file_id": "uuid-string.pdf",
  "extracted_text": "Full text content...",
  "tables": ["<table>...</table>"],  // if extract_tables=true
  "form_data": {"Name": "John Doe"}  // if extract_forms=true
}
```

## 🛠️ Technology Stack

- **Backend Framework**: FastAPI
- **OCR Engine**: EasyOCR (detection & printed text recognition)
- **Handwriting Recognition**: Microsoft TrOCR-Handwritten
- **Table Extraction**: img2table
- **PDF Processing**: pypdfium2 (no Poppler dependency)
- **Frontend**: Vanilla HTML/CSS/JavaScript

## 📁 Project Structure

```
IntelligentDocSys/
├── backend/
│   ├── main.py              # FastAPI application & routes
│   ├── ocr_engine.py        # OCR processing logic
│   ├── requirements.txt     # Python dependencies
│   ├── frontend/
│   │   └── index.html       # Web interface
│   └── uploads/             # Temporary file storage
└── README.md                # This file
```

## ⚙️ Configuration

### OCR Engine Settings

Edit `ocr_engine.py` to customize:
- **GPU Usage**: `OCREngine(use_gpu=True)` (line 19)
- **PDF Render Quality**: `page.render(scale=3)` in `main.py` (higher = better quality, slower)
- **Bounding Box Padding**: `padding = 10` in `ocr_engine.py` (line 109)

### Model Selection

The system uses:
- **Printed Text**: EasyOCR (default)
- **Handwriting**: `microsoft/trocr-base-handwritten` (lazy loaded)
- **Tables**: img2table with EasyOCR backend

## 🐛 Troubleshooting

### Server won't start
- Check if port 8000 is available
- Verify all dependencies are installed: `pip install -r requirements.txt`

### Low OCR accuracy
- For PDFs: Increase render scale in `main.py` (line 68)
- For handwriting: Use `?handwriting=true` parameter
- Ensure images are at least 300 DPI

### Out of memory errors
- Reduce PDF render scale
- Process fewer pages at once
- Disable GPU: `OCREngine(use_gpu=False)`

## 📝 API Documentation

Interactive API documentation is automatically generated and available at:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc

## 🔒 Security Notes

- Files are stored temporarily in the `uploads/` directory
- Consider implementing authentication for production use
- Add rate limiting for public deployments
- Set up CORS policies if accessing from different domains

## 📊 Performance

- **Startup Time**: ~10-15 seconds (model loading)
- **Single Page PDF**: ~2-5 seconds
- **Handwriting Recognition**: ~3-7 seconds per page
- **Table Extraction**: +1-2 seconds per table

## 🤝 Contributing

This project uses:
- Black for code formatting
- Type hints for better code clarity
- REST API best practices

## 📄 License

Educational/Personal use project

## 🙏 Acknowledgments

- EasyOCR for robust text detection
- Microsoft TrOCR for handwriting recognition
- img2table for table structure detection
- FastAPI for the excellent web framework

---

**Need help?** Check the API docs at http://127.0.0.1:8000/docs or review the code comments in `main.py` and `ocr_engine.py`.
