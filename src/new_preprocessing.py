import os
import json
import requests
import re
import nltk
import numpy as np
import pytesseract
import tensorflow as tf
import cv2  # <-- For bounding-box detection
from transformers import pipeline
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from pdf2image import convert_from_path
from PyPDF2 import PdfReader

pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Tokenizer = tf.keras.preprocessing.text.Tokenizer

nltk.download('punkt')

# ---------------------------
# BERT-Only DocumentCompletionModel
# ---------------------------
class DocumentCompletionModel:
    def __init__(self):
        print("Initializing BERT model for fill-mask...")
        self.fallback_model = pipeline(
            "fill-mask",
            model="bert-base-uncased",
            tokenizer="bert-base-uncased"
        )

    def reconstruct_sentence(self, sentence):
        """
        Predict missing words using BERT fill-mask pipeline.
        If multiple [MASK] tokens exist, we fill them one by one.
        """
        tokens = sentence.split()
        # We'll do multiple passes in case there's more than one [MASK].
        while "[MASK]" in tokens:
            idx = tokens.index("[MASK]")
            predictions = self.fallback_model(" ".join(tokens))
            if len(predictions) > 0:
                best_word = predictions[0]["token_str"]
                tokens[idx] = best_word
            else:
                # If no predictions, just remove the mask or replace with something
                tokens[idx] = ""
        return " ".join(tokens).strip()

# Instantiate the BERT-based model
completion_model = DocumentCompletionModel()

def fetch_document_links(base_url):
    """Scrapes the CIA collection page to extract PDF document links."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(base_url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    links = []

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        if href.endswith('.pdf'):
            if not href.startswith('http'):
                href = f"https://www.cia.gov{href}"
            links.append(href)

    return links

def download_documents(links, save_dir="./data/input"):
    """Downloads documents from provided links and saves them locally."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, link in enumerate(links):
        response = requests.get(link)
        response.raise_for_status()
        file_name = os.path.join(save_dir, f"document_{i + 1}.pdf")
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"Saved: {file_name}")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyPDF2."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def clean_text(text):
    """Cleans and preprocesses text for analysis."""
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ")
    return text.strip()

# ------------
# BOUNDING BOX DETECTION
# ------------
def detect_redactions_in_image(pil_image, min_area=3000, color_mode="white"):
    """
    Detects large rectangular regions of uniform color (white or black)
    that might represent redactions.
    :param pil_image: A Pillow Image object of a single PDF page.
    :param min_area: Minimum bounding box area to consider a region as a redaction
    :param color_mode: "white" or "black" for detecting whited-out or blacked-out bars
    :return: A list of bounding boxes in (x, y, w, h) format for suspected redactions
    """
    cv_image = np.array(pil_image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    if color_mode == "white":
        # invert the image if searching for white blocks
        gray = cv2.bitwise_not(gray)

    # threshold to get large uniform areas
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    redacted_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area >= min_area:
            redacted_boxes.append((x, y, w, h))
    return redacted_boxes

def get_tesseract_bounding_boxes(pil_image):
    """
    Returns Tesseract word-level bounding boxes from a Pillow image.
    Each box is a dict: { 'text': str, 'left': x, 'top': y, 'width': w, 'height': h }
    """
    data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
    boxes = []
    n = len(data['level'])
    for i in range(n):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        if conf > 20 and text:
            left, top = data['left'][i], data['top'][i]
            width, height = data['width'][i], data['height'][i]
            boxes.append({
                "text": text,
                "left": left,
                "top": top,
                "width": width,
                "height": height
            })
    return boxes

def overlap_area(b1, b2):
    """
    Returns the overlapping area between two bounding boxes b1, b2
    Each b is (x, y, w, h)
    """
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    # convert w,h to x2,y2
    r1x2 = x1 + w1
    r1y2 = y1 + h1
    r2x2 = x2 + w2
    r2y2 = y2 + h2

    overlap_x1 = max(x1, x2)
    overlap_y1 = max(y1, y2)
    overlap_x2 = min(r1x2, r2x2)
    overlap_y2 = min(r1y2, r2y2)

    if overlap_x2 > overlap_x1 and overlap_y2 > overlap_y1:
        return (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
    return 0

def is_inside_redaction(word_box, redaction_boxes):
    """
    Check if the recognized word_box overlaps with any redaction box significantly.
    We'll do a simple approach: if there's any overlap area > 50% of the word_box area, we call it inside.
    """
    x, y, w, h = word_box
    word_area = w * h
    for rb in redaction_boxes:
        ovl = overlap_area(word_box, rb)
        if ovl > 0.5 * word_area:
            return True
    return False

def build_page_text_with_redactions(pil_img, redaction_boxes):
    """
    Build a text string from Tesseract word bounding boxes, skipping words inside redaction
    and adding [MASK] around each redaction region.
    - We'll sort recognized words by reading order (top->down, left->right).
    - Whenever we enter a redaction region, we place [MASK].
    - We skip words that are inside the redaction region, and then place [MASK] after we exit that region.
    """
    tess_boxes = get_tesseract_bounding_boxes(pil_img)
    if not tess_boxes:
        return ""

    # sort boxes by y, then x to approximate reading order
    tess_boxes.sort(key=lambda b: (b["top"], b["left"]))

    output_words = []
    in_redaction = False

    for tb in tess_boxes:
        text = tb["text"]
        x, y, w, h = tb["left"], tb["top"], tb["width"], tb["height"]

        if is_inside_redaction((x, y, w, h), redaction_boxes):
            # inside a redacted region
            if not in_redaction:
                # we are just entering a redacted region
                output_words.append("[MASK]")  # start block
                in_redaction = True
            # skip the word
        else:
            # outside a redacted region
            if in_redaction:
                # we just exited the region
                output_words.append("[MASK]")
                in_redaction = False
            # add this recognized word to the text
            output_words.append(text)

    # if we end in a redacted region, close it
    if in_redaction:
        output_words.append("[MASK]")

    return " ".join(output_words)

def is_redaction(x, y, w, h, tesseract_boxes):
    """
    Checks if bounding box (x, y, w, h) is truly a redaction,
    i.e. there's no recognized text box inside it
    (or the recognized text is small overlap).
    We'll do the same area overlap approach.
    """
    # If we find any recognized text that overlaps significantly with this box, it's not "blank"
    rx_area = w * h
    for tb in tesseract_boxes:
        tx, ty, tw, th = tb["left"], tb["top"], tb["width"], tb["height"]
        # check overlap
        ovl = overlap_area((x, y, w, h), (tx, ty, tw, th))
        # if more than 5% overlap, let's call it "not blank"
        if ovl > 0.05 * (tw * th):
            # means there's recognized text
            return False
    return True

def process_documents_bounding_box(
    doc_dir="./data/input",
    color_mode="white",
    min_area=3000
):
    """
    Processes PDF documents, identifies bounding-box redactions using OpenCV,
    inserts [MASK] tags around redacted blocks, and collects everything
    into a dataset list of dicts.
    """
    dataset = []
    
    for file_name in os.listdir(doc_dir):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(doc_dir, file_name)
            pages = convert_from_path(file_path)

            page_redactions = []
            combined_text_pages = []
            
            for page_index, pil_img in enumerate(pages):
                # 1) Detect suspicious rectangles
                redaction_boxes = detect_redactions_in_image(
                    pil_img, min_area=min_area, color_mode=color_mode
                )

                # 2) We'll label a region as "confirmed redaction" if truly blank
                #    i.e. there's no recognized text in it.
                tess_boxes = get_tesseract_bounding_boxes(pil_img)
                confirmed_redactions = []
                for (x, y, w, h) in redaction_boxes:
                    if is_redaction(x, y, w, h, tess_boxes):
                        confirmed_redactions.append((x, y, w, h))

                # Rebuild final page text with [MASK] around redacted blocks
                page_text_with_masks = build_page_text_with_redactions(
                    pil_img, confirmed_redactions
                )
                combined_text_pages.append(page_text_with_masks)

                page_redactions.append({
                    "page_index": page_index,
                    "redaction_boxes": confirmed_redactions
                })

            # Combine all pages text
            masked_text = "\n".join(combined_text_pages)

            # The PDFReader extracted text (non-ocr) for reference
            extracted_text = extract_text_from_pdf(file_path)

            # Clean
            cleaned_text = clean_text(masked_text)

            dataset.append({
                "file_name": file_name,
                "raw_text": extracted_text,
                "masked_text": masked_text,
                "cleaned_text": cleaned_text,
                "sentences": sent_tokenize(cleaned_text),
                "bounding_box_redactions": page_redactions
            })

    return dataset

def reconstruct_redacted_text(text):
    """
    Runs BERT fill-mask on the entire text that has multiple [MASK] blocks
    We do sentence-by-sentence to keep it simpler.
    """
    sentences = sent_tokenize(text)
    reconstructed = []
    for s in sentences:
        if "[MASK]" in s:
            replaced = completion_model.reconstruct_sentence(s)
            reconstructed.append(replaced)
        else:
            reconstructed.append(s)
    return " ".join(reconstructed)

def save_dataset(dataset, output_file="./data/processed/processed_dataset.json"):
    """Saves the processed dataset as a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    base_url = input("Enter the URL of the CIA collection page: ").strip()
    document_links = fetch_document_links(base_url)
    print(f"Found {len(document_links)} document links.")

    download_documents(document_links)
    
    # bounding-box-based detection & insertion of [MASK]
    processed_dataset = process_documents_bounding_box(
        doc_dir="./data/input",
        color_mode="white",   # or "black" if you're dealing with black bars
        min_area=3000
    )

    # Optional step: run fill-mask reconstruction on the entire text
    # for item in processed_dataset:
    #     item["reconstructed_text"] = reconstruct_redacted_text(item["masked_text"])

    save_dataset(processed_dataset)
    print("Processing complete!")
