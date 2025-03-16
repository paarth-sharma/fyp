#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import re
import json
import random
import wikipedia
import nltk
nltk.download('punkt', quiet=True)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler

from transformers import (
    BertTokenizer,
    BertForPreTraining,
    pipeline,
    AdamW
)

###############################################################################
# Global Config
###############################################################################
DEVICE = "cuda:0"  # or "set 'cpu' if no extrernal gpu detected"
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_pretrain_model = BertForPreTraining.from_pretrained("bert-base-uncased")
bert_pretrain_model.to(DEVICE)

# Domain-specific terms we always want to mask
DOMAIN_TERMS = {"bar", "lev", "idf", "sinai", "suez", "israeli"}
JSON_INDEX = {}
random.seed(42)


###############################################################################
# Summarization
###############################################################################
try:
    bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
except Exception:
    bart_summarizer = None

def summarize_if_needed(text_in, max_bert_length=512):
    if bart_summarizer is None:
        return text_in
    tokens = bert_tokenizer.encode(text_in, add_special_tokens=False)
    if len(tokens) <= max_bert_length:
        return text_in
    summary_out = bart_summarizer(text_in, max_length=150, min_length=40, do_sample=False)
    return summary_out[0]["summary_text"]


###############################################################################
# Prepare (Sentence A, Sentence B) Pairs for Next Sentence Prediction
###############################################################################
def make_sentence_pairs(original_sents):
    """
    50%: (sentA, nextSent) label=1
    50%: (sentA, randomSent) label=0
    """
    pairs = []
    for i in range(len(original_sents) - 1):
        sentA = original_sents[i]
        if random.random() < 0.5:
            sentB = original_sents[i+1]
            label = 1
        else:
            # random pick
            rand_idx = random.randint(0, len(original_sents) - 1)
            sentB = original_sents[rand_idx]
            label = 0
        pairs.append((sentA, sentB, label))
    return pairs

###############################################################################
# Create Pretraining Examples with "Hard Force" Domain Term Masking
###############################################################################
def create_pretraining_examples(pairs, tokenizer, max_length=512, mask_prob=0.15):
    """
    We'll do:
       - always mask domain terms if encountered
       - for other tokens: 80% [MASK], 10% random, 10% same
    """
    examples = []
    vocab_keys = list(tokenizer.vocab.keys())

    for (sentA, sentB, label) in pairs:
        tokensA = tokenizer.tokenize(sentA)
        tokensB = tokenizer.tokenize(sentB)
        while len(tokensA) + len(tokensB) > (max_length - 3):
            if len(tokensA) > len(tokensB):
                tokensA.pop()
            else:
                tokensB.pop()

        tokens = ["[CLS]"] + tokensA + ["[SEP]"] + tokensB + ["[SEP]"]
        seg_ids = [0]*(len(tokensA)+2) + [1]*(len(tokensB)+1)

        input_tokens = []
        masked_labels = []

        for tok in tokens:
            if tok in ["[CLS]", "[SEP]"]:
                input_tokens.append(tok)
                masked_labels.append(-100)  # ignore
                continue

            lowered = tok.lower()
            # force mask domain terms
            if any(dterm in lowered for dterm in DOMAIN_TERMS):
                # always mask
                input_tokens.append("[MASK]")
                masked_labels.append(tokenizer.vocab.get(tok, tokenizer.vocab["[UNK]"]))
            else:
                # normal random strategy
                if random.random() < mask_prob:
                    # 80-10-10
                    r = random.random()
                    if r < 0.8:
                        chosen = "[MASK]"
                    elif r < 0.9:
                        chosen = random.choice(vocab_keys)
                    else:
                        chosen = tok
                    input_tokens.append(chosen)
                    masked_labels.append(tokenizer.vocab.get(tok, tokenizer.vocab["[UNK]"]))
                else:
                    input_tokens.append(tok)
                    masked_labels.append(-100)

        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        ex = {
            "input_ids": input_ids,
            "token_type_ids": seg_ids,
            "masked_lm_labels": masked_labels,
            "next_sentence_label": label
        }
        examples.append(ex)
    return examples

###############################################################################
# Data Collation
###############################################################################
class PretrainDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_pretrain(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    if max_len > 512:
        max_len = 512

    input_ids_list = []
    token_type_ids_list = []
    masked_lm_labels_list = []
    nsp_labels_list = []
    attention_masks_list = []

    for ex in batch:
        inp = ex["input_ids"][:max_len]
        seg = ex["token_type_ids"][:max_len]
        lbl = ex["masked_lm_labels"][:max_len]
        amask = [1]*len(inp)

        while len(inp) < max_len:
            inp.append(0)
            seg.append(0)
            lbl.append(-100)
            amask.append(0)

        input_ids_list.append(inp)
        token_type_ids_list.append(seg)
        masked_lm_labels_list.append(lbl)
        nsp_labels_list.append(ex["next_sentence_label"])
        attention_masks_list.append(amask)

    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    token_type_ids = torch.tensor(token_type_ids_list, dtype=torch.long)
    masked_lm_labels = torch.tensor(masked_lm_labels_list, dtype=torch.long)
    attention_masks = torch.tensor(attention_masks_list, dtype=torch.long)
    nsp_labels = torch.tensor(nsp_labels_list, dtype=torch.long)

    return {
        "input_ids": input_ids,
        "token_type_ids": token_type_ids,
        "attention_mask": attention_masks,
        "labels": masked_lm_labels,
        "next_sentence_label": nsp_labels
    }

###############################################################################
# Pre-train the BERT Model with Hard Domain Masking + NSP
###############################################################################
def pretrain_bert_domain(original_sents, epochs=5, batch_size=8, max_length=128, lr=2e-5):
    pairs = make_sentence_pairs(original_sents)
    examples = create_pretraining_examples(pairs, bert_tokenizer, max_length=max_length, mask_prob=0.15)
    dataset = PretrainDataset(examples)
    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_pretrain)

    optimizer = AdamW(bert_pretrain_model.parameters(), lr=lr)
    bert_pretrain_model.train()

    for ep in range(epochs):
        total_loss = 0
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(DEVICE)
            token_type_ids = batch["token_type_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            masked_labels = batch["labels"].to(DEVICE)
            nsp_labels = batch["next_sentence_label"].to(DEVICE)

            outputs = bert_pretrain_model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=masked_labels,
                next_sentence_label=nsp_labels
            )
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 20 == 0:
                print(f"Epoch {ep+1} | Step {step} | Loss={loss.item():.4f}")
        print(f"Epoch {ep+1} completed | Average Loss: {total_loss/len(loader):.4f}")

    bert_pretrain_model.eval()

###############################################################################
# Building JSON Index & Attempting Direct Replacement
###############################################################################
def build_json_index(original_sentences):
    global JSON_INDEX
    JSON_INDEX = {}
    for idx, sent in enumerate(original_sentences):
        words = sent.lower().split()
        for w in words:
            if w not in JSON_INDEX:
                JSON_INDEX[w] = []
            JSON_INDEX[w].append(idx)

def find_potential_matches(redacted_text, original_sentences, max_candidates=5):
    global JSON_INDEX
    tokens = redacted_text.lower().split()
    overlap = {}
    for t in tokens:
        if t in JSON_INDEX:
            for i in JSON_INDEX[t]:
                overlap[i] = overlap.get(i, 0) + 1

    sorted_idx = sorted(overlap.keys(), key=lambda x: overlap[x], reverse=True)
    out = []
    for sid in sorted_idx[:max_candidates]:
        out.append(original_sentences[sid])
    return out

def attempt_exact_replacement(redacted_text, original_sentences):
    patt = re.escape(redacted_text).replace("\\[REDACTED\\]", "(.*?)")
    rx = re.compile(patt, re.IGNORECASE)
    cands = find_potential_matches(redacted_text, original_sentences, max_candidates=15)
    for c in cands:
        m = rx.match(c)
        if m:
            rep = m.group(1)
            if rep and len(rep.strip())>0:
                print(f"✓ Direct match in JSON: '{rep}'")
                replaced = redacted_text.replace("[REDACTED]", rep)
                return True, replaced
    return False, redacted_text

###############################################################################
# Wikipedia & Corpus Context
###############################################################################
def gather_wiki_context(search_terms, max_chars=1000):
    text = ""
    for term in search_terms:
        try:
            s = wikipedia.summary(term, sentences=3)
            if len(text)+len(s)<max_chars:
                text += " " + s
            else:
                break
        except:
            pass
    return text.strip()

def gather_corpus_context(redacted_text, original_sents, max_chars=1000):
    # same as your gather corpus logic
    words = [w.lower() for w in redacted_text.split() if w != "[REDACTED]" and len(w)>4]
    rel = []
    pattern = re.compile(redacted_text.replace("[REDACTED]", "(.*?)"), re.IGNORECASE)
    for s in original_sents:
        if pattern.match(s):
            rel.append(s)
    if not rel:
        for s in original_sents:
            lw = s.lower()
            if any(w in lw for w in words):
                rel.append(s)
    joined = " ".join(rel)
    if len(joined)>max_chars:
        joined = joined[:max_chars]+"..."
    return joined

###############################################################################
# Reranking with Extra Bonus for Domain Terms
###############################################################################
def rerank_predictions(candidates, original_sents):
    # candidates = [(token, prob), ...]
    # we want to push "bar lev" strongly if it appears
    # We'll just do a big bonus if the candidate is in domain terms or in the JSON
    all_words = set(w.lower() for w in " ".join(original_sents).split())

    scored = []
    for (tok, prob) in candidates:
        low = tok.lower()
        # if domain
        domain_bonus = 5.0 if any(dterm in low for dterm in DOMAIN_TERMS) else 0.0
        # if in JSON
        in_json = 3.0 if low in all_words else 0.0
        final = prob*10 + domain_bonus + in_json
        scored.append((tok, prob, final))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored

###############################################################################
# Iterative Decensoring
###############################################################################
def iterative_decensoring(redacted_text, original_sents, wiki_terms=None, epochs=3, top_k=5):
    # Step 1: direct from JSON
    found, replaced = attempt_exact_replacement(redacted_text, original_sents)
    if found:
        return replaced

    combined_text = redacted_text

    # gather context
    context_str = ""
    if wiki_terms:
        wiki = gather_wiki_context(wiki_terms)
        if wiki:
            context_str += "WikiCtx: " + wiki + "\n\n"
    corp = gather_corpus_context(redacted_text, original_sents)
    context_str += "CorpCtx: " + corp + "\n\n"

    new_input = f"{context_str}\nRedactedText: {combined_text}"

    for ep in range(epochs):
        if "[REDACTED]" not in new_input:
            print(f"No more placeholders at epoch {ep+1}.")
            break
        short = summarize_if_needed(new_input, max_bert_length=450)
        # replace 1
        masked_text = short.replace("[REDACTED]", "[MASK]", 1)

        enc = bert_tokenizer.encode_plus(masked_text, return_tensors="pt")
        input_ids = enc["input_ids"].to(DEVICE)

        # forward
        with torch.no_grad():
            out = bert_pretrain_model(input_ids=input_ids)
        logits = out.prediction_logits
        # find [MASK]
        mask_idxs = (input_ids == bert_tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
        if mask_idxs.size(0)==0:
            break
        idx0 = mask_idxs[0].item()
        mask_logits = logits[0, idx0, :]
        probs = F.softmax(mask_logits, dim=-1)
        val, idxs = probs.topk(top_k+10)

        # filter
        cands = []
        special = {"[mask]", "[pad]", "[cls]", "[sep]", "[unk]", "", " "}
        for v, i in zip(val, idxs):
            token = bert_tokenizer.convert_ids_to_tokens(i.item())
            p = float(v.item())
            if token.lower() not in special and len(token)>1 and not token.startswith("##"):
                cands.append((token, p))
                if len(cands)>=top_k:
                    break

        if not cands:
            cands = [("unknown", 1.0)]

        # rerank
        final_candidates = rerank_predictions(cands, original_sents)
        best = final_candidates[0][0]
        new_text = masked_text.replace("[MASK]", best, 1)

        print(f"Epoch {ep+1} => Best token '{best}' finalscore={final_candidates[0][2]:.4f}")
        snippet = new_text.split("RedactedText: ")[-1]
        print(f"Current Partial:\n{snippet}\n{'-'*40}")

        new_input = new_text

    if "RedactedText: " in new_input:
        final_text = new_input.split("RedactedText: ")[-1]
    else:
        final_text = new_input
    return final_text


###############################################################################
# MAIN
###############################################################################
def main_pipeline(json_path="./data/processed/document_1_processed.json",
                  wiki_list=["Suez Canal","Bar Lev"],
                  user_redacted="Along the Suez Canal, the [REDACTED] defense line has done...",
                  epochs=5, top_k=5):
    print("\n=== DECENSORING PIPELINE with Hard Domain Masking & Weighted Rerank ===\n")

    # load JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        original_sents = data["original_sentences"]
        censored_sents = data["censored_sentences"]
        print(f"Loaded from {json_path}: {len(original_sents)} original, {len(censored_sents)} censored")
    except:
        print("JSON loading error, fallback to defaults.")
        original_sents = [
            "Along the Suez Canal, the Bar Lev defense line has done the job of tripping the Egyptian offensive.",
            "IDF forces were present along the line as well."
        ]
        censored_sents = [
            "Along the Suez Canal, the [REDACTED] defense line has done the job of tripping the Egyptian offensive."
        ]

    # pretrain BERT w domain data
    print("\n--- PHASE 1: Pretrain BERT (MLM+NSP) w domain terms forced masked ---")
    pretrain_bert_domain(original_sents, epochs=5, batch_size=8, max_length=128, lr=2e-5)
    print("Pretraining done.\n")

    # build index
    build_json_index(original_sents)

    # pick text
    if user_redacted:
        decensor_me = user_redacted
    elif censored_sents:
        decensor_me = random.choice(censored_sents)
    else:
        decensor_me = "Along the Suez Canal, the [REDACTED] was unstoppable."

    print(f"Decensoring: \"{decensor_me}\"\n")

    final = iterative_decensoring(decensor_me, original_sents, wiki_terms=wiki_list, epochs=epochs, top_k=top_k)

    print("\n=== FINAL DECENSORED TEXT ===")
    print(f"Original:\n{decensor_me}\n\nDecensored:\n{final}\n")


if __name__=="__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--json_path", type=str, default="./data/processed/document_1_processed.json")
    ap.add_argument("--wiki_terms", type=str, default="Suez Canal,Bar Lev")
    ap.add_argument("--input", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--top_k", type=int, default=5)
    args = ap.parse_args()

    wiki_list = [w.strip() for w in args.wiki_terms.split(",")]
    main_pipeline(
        json_path=args.json_path,
        wiki_list=wiki_list,
        user_redacted=args.input,
        epochs=args.epochs,
        top_k=args.top_k
    )
