import os
import re
import json
import requests
import cv2
import numpy as np
import pytesseract
import spacy
from bs4 import BeautifulSoup
from pdf2image import convert_from_path

#######################################################
# 1. FETCH + DOWNLOAD DOCUMENTS
#######################################################

def fetch_document_links(base_url):
    """
    Scrapes the CIA collection page (base_url) to extract PDF document links.
    Only appends .pdf links.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(base_url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    links = []

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.endswith('.pdf'):
            # If the link is relative, prepend domain
            if not href.startswith('http'):
                href = f"https://www.cia.gov{href}"
            links.append(href)

    return links

def download_documents(links, save_dir="./data/input"):
    """
    Downloads .pdf documents from the provided links and saves them locally
    in save_dir. Filenames are enumerated: "document_1.pdf", etc.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, link in enumerate(links):
        response = requests.get(link)
        response.raise_for_status()
        file_name = os.path.join(save_dir, f"document_{i + 1}.pdf")
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"Saved: {file_name}")

#######################################################
# 2. PROCESS DOCUMENTS (OCR + REDACTION DETECTION + SPACY NER)
#######################################################

# Make sure you've installed a suitable spacy model:
# python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

def detect_and_remove_redactions(image):
    """
    Attempts to detect large rectangular redactions in 'image'
    and remove them by whiting out the rectangle.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold to find large black/white boxes
    # Adjust blockSize, C to your PDF's typical redactions
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=35,  # play with this
        C=15           # play with this
    )
    
    # Find contours
    contours, hierarchy = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Filter out small or large areas (skip entire page edges or tiny lines)
    h, w = image.shape[:2]
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch

        # skip small specks
        if area < 5000:
            continue
        # skip massive area (like entire page)
        if area > 0.5 * w * h:
            continue
        
        # White out region
        cv2.rectangle(image, (x, y), (x+cw, y+ch), (255, 255, 255), -1)

    return image

def extract_text_from_pdf(pdf_path, dpi=300):
    """
    Converts PDF pages to images, tries to remove redactions,
    then OCRs each page. Returns a list of text strings (one per page).
    """
    pages = convert_from_path(pdf_path, dpi=dpi)
    ocr_texts = []

    for i, pil_image in enumerate(pages):
        # Convert PIL image to OpenCV format
        cv2_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Detect & remove large redaction blocks
        cleaned_img = detect_and_remove_redactions(cv2_img)
        
        # OCR with pytesseract
        text = pytesseract.image_to_string(cleaned_img)
        
        # Basic cleaning
        text = text.replace('\x0c', ' ').strip()
        ocr_texts.append(text)

    return ocr_texts

def split_into_sentences_spacy(text):
    """
    Use spaCy's sentence segmentation to split text into coherent English sentences,
    removing short or non-alphabetic lines.
    """
    doc = nlp(text)
    sentences = []
    for sent in doc.sents:
        s = sent.text.strip()
        # Filter out extremely short or non-alphabetic lines
        if len(s) > 5 and re.search('[a-zA-Z]', s):
            sentences.append(s)
    return sentences

def censor_named_entities(sentence):
    """
    Censor sensitive NER tags in sentence: PERSON, ORG, GPE, etc.
    Replace them with [REDACTED].
    """
    doc = nlp(sentence)
    censored_sentence = ""
    current_idx = 0

    # Decide which entity labels to treat as sensitive
    sensitive_labels = {"PERSON", "ORG", "GPE"}

    for ent in doc.ents:
        if ent.label_ in sensitive_labels:
            censored_sentence += sentence[current_idx:ent.start_char]
            censored_sentence += "[REDACTED]"
            current_idx = ent.end_char
    
    censored_sentence += sentence[current_idx:]
    return censored_sentence

def process_pdf(pdf_path, output_json_path="./data/output/output.json"):
    """
    Main pipeline for a single PDF:
    1. OCR the PDF (with redaction-block detection).
    2. Combine page texts, split into sentences.
    3. For each sentence, produce a censored version.
    4. Write as JSON with "original_sentences" and "censored_sentences".
    """
    ocr_pages = extract_text_from_pdf(pdf_path)
    full_text = "\n".join(ocr_pages)

    # Split into sentences
    sentences = split_into_sentences_spacy(full_text)

    # Censor each
    censored_sentences = [censor_named_entities(s) for s in sentences]

    # Create final structure
    data = {
        "original_sentences": sentences,
        "censored_sentences": censored_sentences
    }

    # Ensure output folder
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Processed {pdf_path} -> {output_json_path}")

#######################################################
# 3. Putting It All Together
#######################################################

def main():
    # 1. Provide the base CIA page containing PDF links
    base_url = "https://www.cia.gov/readingroom/document/51112a4b993247d4d839452a"  # example
    pdf_save_dir = "./data/input"
    output_dir = "./data/processed"

    # 2. Fetch PDF links
    links = fetch_document_links(base_url)
    print(f"Found {len(links)} PDF links.")

    # 3. Download them
    download_documents(links, save_dir=pdf_save_dir)

    # 4. Process each PDF
    pdf_files = [f for f in os.listdir(pdf_save_dir) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_save_dir, pdf_file)

        # Create a corresponding output file, e.g. "document_1.json"
        base_name = os.path.splitext(pdf_file)[0]
        json_path = os.path.join(output_dir, f"{base_name}_processed.json")

        process_pdf(pdf_path, json_path)

if __name__ == "__main__":
    main()
