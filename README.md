## Setup

1. Install these 2 programs:
    - [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/), which is required by ```pdf2image``` to convert PDFs into images for OCR processing. Extract the ZIP file (e.g., ```poppler-23.11.0```). Copy the extracted folder path and add it to your system path variable (e.g., ```C:\poppler-23.11.0\Library\bin```).
    - [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki), Run the installer and remember the installation path (default: ```C:\Program Files\Tesseract-OCR\```). During installation, select Additional Language Data (if needed for multilingual PDFs).

2. Run the commands below to install libraries for the project

    1. Create a virtual environment and activate it
    ```
    python -m venv venv
    .\venv\Scripts\activate
    ```

    2. Install pytorch for the GPU (default options use the CPU)
    ```
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

    3. Install the spaCy model that will help us tag entities
    ```
    python -m spacy download en_core_web_sm
    ```

    4. Install python libraries
    ```
    pip install numpy nltk requests beautifulsoup4 PyPDF2 torch pytesseract pdf2image tensorflow spacy
    ```

3. Obtain the api keys for the following
    1. [LANGSMITH_API_KEY](https://smith.langchain.com/)create a workspace and a project and the constant names will be generated for you.
    2. [API_KEY](https://developers.google.com/custom-search/v1/introduction)for google search json api.
    3. [SEARCH_ENGINE_ID](https://programmablesearchengine.google.com/), create a custom web engine and then copy its id.

4. Make a ```.env``` file to manage all sensitive api keys which this will reside in the root of the project folder
```.env
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY= ""
LANGSMITH_PROJECT= ""
API_KEY = ""
SEARCH_ENGINE_ID = ""
```
## How to
1. copy page from: the cia website with link as: https://www.cia.gov/readingroom/document/51112a4a993247d4d8394487 and enter the whole link in the terminal.
2. run the script with the following command(s)
```
# template command and arguments
python .\context-bridge.py --json_path "./data/processed/<processed_json_file>" --wiki_terms "<comma seperated 2 or 3 terms>" --input "< sentence with atleast 1 [REDACTED] tag in it>"

# example with all parameters
python .\context-bridge.py --json_path "./data/processed/document_1_processed.json" --wiki_terms "Sinai, Suez canal" --input "In the [REDACTED], the overall level."
```

## Resources
### Papers


