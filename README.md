### Setup

- Install these 2 programs:
    - [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/), which is required by ```pdf2image``` to convert PDFs into images for OCR processing. Extract the ZIP file (e.g., ```poppler-23.11.0```). Copy the extracted folder path and add it to your system path variable (e.g., ```C:\poppler-23.11.0\Library\bin```).
    - [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki), Run the installer and remember the installation path (default: ```C:\Program Files\Tesseract-OCR\```). During installation, select Additional Language Data (if needed for multilingual PDFs).

- run command below to install libraries for the project
```
#create a virtual environment and activate it
python -m venv venv
.\venv\Scripts\activate

#install python libraries
pip install numpy nltk requests beautifulsoup4 PyPDF2 torch pytesseract pdf2image tensorflow spacy
```