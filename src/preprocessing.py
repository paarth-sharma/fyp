import os
import json
import requests
import re
import nltk
import numpy as np
import pytesseract
import tensorflow as tf
from transformers import pipeline
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from pdf2image import convert_from_path
from PyPDF2 import PdfReader

load_model = tf.keras.models.load_model
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Tokenizer = tf.keras.preprocessing.text.Tokenizer

nltk.download('punkt')

# Set up reproducibility
seed = 42
np.random.seed(seed)

# LSTM Model Configuration
MAX_VOCAB = 50000
MAX_SEQ_LENGTH = 128
EMBEDDING_DIM = 256
LSTM_UNITS = 512
MODEL_PATH = "models/lstm_declassification.h5"
TOKENIZER_PATH = "models/tokenizer.json"

class DocumentCompletionModel:
    def __init__(self):
        self.tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
        self.model = None
        self.fallback_model = None  # Initialize here
        
        try:
            # Load LSTM model
            self.model = load_model(MODEL_PATH)
            with open(TOKENIZER_PATH, 'r') as f:
                tokenizer_config = json.load(f)
                self.tokenizer = Tokenizer.from_json(tokenizer_config)
            print("Successfully loaded LSTM model")
        except Exception as e:
            print(f"LSTM load failed: {e}")
            self._init_fallback_model()
    
    def _init_fallback_model(self):
        """Initialize BERT fallback only if needed"""
        print("Initializing BERT fallback model...")
        self.fallback_model = pipeline("fill-mask", 
                                      model="bert-base-uncased",
                                      tokenizer="bert-base-uncased")

    def reconstruct_sentence(self, sentence):
        """Predict missing words using LSTM with fallback to BERT"""
        if not self.model:
            return self._fallback_reconstruction(sentence)
            
        tokens = word_tokenize(sentence)
        masked_indices = [i for i, word in enumerate(tokens) if word == "[MASK]"]
        
        if not masked_indices:
            return sentence
            
        try:
            seq = self.tokenizer.texts_to_sequences([' '.join(tokens)])
            padded_seq = pad_sequences(seq, maxlen=MAX_SEQ_LENGTH, padding='post')
            predictions = self.model.predict(padded_seq)[0]
            
            for idx in masked_indices:
                if idx < len(predictions):
                    predicted_word = self.tokenizer.index_word.get(
                        np.argmax(predictions[idx]), 
                        "[UNKNOWN]"
                    )
                    tokens[idx] = predicted_word
            return ' '.join(tokens)
        except Exception as e:
            print(f"LSTM prediction failed: {e}")
            return self._fallback_reconstruction(sentence)

    def _fallback_reconstruction(self, sentence):
        """Use BERT as fallback if LSTM fails"""
        predictions = self.fallback_model(sentence)
        return predictions[0]["sequence"] if predictions else sentence

# Initialize document completion model
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

def download_documents(links, save_dir="cia_documents"):
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

def extract_text_with_ocr(pdf_path):
    """Converts PDF pages to images and extracts text using OCR."""
    images = convert_from_path(pdf_path)
    ocr_text = "\n".join([pytesseract.image_to_string(img) for img in images])
    return ocr_text.strip()

def reconstruct_text(ocr_text, extracted_text):
    """Reconstructs missing text using LSTM model with OCR context."""
    if not ocr_text or len(ocr_text) < len(extracted_text):
        sentences = sent_tokenize(extracted_text)
        reconstructed_sentences = [
            completion_model.reconstruct_sentence(sentence)
            if "[MASK]" in sentence else sentence
            for sentence in sentences
        ]
        return " ".join(reconstructed_sentences)
    return extracted_text

def clean_text(text):
    """Cleans and preprocesses text for analysis."""
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\n", " ")
    return text

def process_documents(doc_dir="cia_documents"):
    """Processes all downloaded documents and reconstructs text."""
    dataset = []
    
    for file_name in os.listdir(doc_dir):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(doc_dir, file_name)
            
            extracted_text = extract_text_from_pdf(file_path)
            ocr_text = extract_text_with_ocr(file_path)
            reconstructed_text = reconstruct_text(ocr_text, extracted_text)
            cleaned_text = clean_text(reconstructed_text)

            dataset.append({
                "file_name": file_name,
                "raw_text": extracted_text,
                "ocr_text": ocr_text,
                "reconstructed_text": reconstructed_text,
                "cleaned_text": cleaned_text,
                "sentences": sent_tokenize(cleaned_text)
            })

    return dataset

def save_dataset(dataset, output_file="processed_dataset.json"):
    """Saves the processed dataset as a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    base_url = input("Enter the URL of the CIA collection page: ").strip()
    document_links = fetch_document_links(base_url)
    print(f"Found {len(document_links)} document links.")
    
    download_documents(document_links)
    processed_dataset = process_documents()
    save_dataset(processed_dataset)
    print("Processing complete!")