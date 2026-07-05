import spacy
import torch
import json
import pickle
import numpy as np
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from scipy.sparse import csr_matrix
import time
import os

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load English model
print("Loading SpaCy model...")
nlp = spacy.load("en_core_web_sm")

# Load RoBERTa model (có thể dùng từ cache hoặc download)
print("Loading RoBERTa model...")
try:
    # Thử load từ thư mục local nếu có
    model = RobertaModel.from_pretrained("./roberta-base/")
    tokenizer = RobertaTokenizer.from_pretrained("./roberta-base/")
except:
    # Nếu không thì download
    print("Downloading RoBERTa-base model...")
    model = RobertaModel.from_pretrained("roberta-base")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # Lưu lại để dùng sau
    model.save_pretrained("./roberta-base/")
    tokenizer.save_pretrained("./roberta-base/")

model = model.to(device)
model.eval()

def load_jsonl_data(file_path):
    """Đọc file JSONL thay vì JSON"""
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            texts.append(data['text'])
            # Label: 0=human, 1=AI (theo dữ liệu của bạn)
            label = data['label']
            labels.append(label)
    return texts, labels

def build_graph_from_texts(texts, labels):
    """Xây dựng đồ thị từ list texts và labels"""
    start_time = time.time()
    
    y = torch.tensor(labels, dtype=torch.float32)
    all_token_embeddings = []
    all_edge_index = []
    
    for idx, text in enumerate(tqdm(texts, desc="Processing texts")):
        try:
            # SpaCy parsing
            doc = nlp(text[:100000])  # Giới hạn độ dài
            tokenized_sentence = [token.text for token in doc]
            
            if len(tokenized_sentence) == 0:
                continue
            
            # Get RoBERTa embeddings with chunking for long texts
            max_length = 512
            chunks = [tokenized_sentence[i:i+max_length] for i in range(0, len(tokenized_sentence), max_length)]
            chunk_outputs = []
            
            for chunk in chunks:
                # Convert tokens to ids
                token_ids = tokenizer.convert_tokens_to_ids(chunk)
                if len(token_ids) == 0:
                    continue
                    
                input_ids = torch.tensor(token_ids).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_ids)
                
                last_hidden_states = output.last_hidden_state
                token_embeddings = last_hidden_states[0]  # [seq_len, 768]
                chunk_outputs.append(token_embeddings)
            
            if len(chunk_outputs) == 0:
                continue
                
            token_embeddings = torch.cat(chunk_outputs, dim=0)
            
            # Trim to match number of tokens
            if len(token_embeddings) > len(tokenized_sentence):
                token_embeddings = token_embeddings[:len(tokenized_sentence)]
            elif len(token_embeddings) < len(tokenized_sentence):
                # Pad if needed
                pad_size = len(tokenized_sentence) - len(token_embeddings)
                pad_embeddings = torch.zeros(pad_size, token_embeddings.shape[1]).to(device)
                token_embeddings = torch.cat([token_embeddings, pad_embeddings], dim=0)
            
            all_token_embeddings.append(token_embeddings.cpu())
            
            # Build dependency edges
            node_relations = []
            for word in doc:
                node_relations.append([word.i, word.head.i])
                # Add self-loop (as in paper: A = A + I)
                # node_relations.append([word.i, word.i])  # Uncomment if needed
            
            edge0 = [edge[0] for edge in node_relations]
            edge1 = [edge[1] for edge in node_relations]
            
            # Ensure indices are within bounds
            max_idx = len(tokenized_sentence) - 1
            edge0 = [min(e, max_idx) for e in edge0]
            edge1 = [min(e, max_idx) for e in edge1]
            
            edge_index = torch.tensor([edge0, edge1], dtype=torch.long)
            all_edge_index.append(edge_index)
            
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            print(f"Text preview: {text[:200]}...")
            continue
    
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    print(f"Successfully processed {len(all_token_embeddings)}/{len(texts)} samples")
    
    return all_token_embeddings, all_edge_index, y

def save_pkl(file_name, all_token_embeddings, all_edge_index, y, output_dir="graph_data"):
    """Save graph data to pickle file"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{file_name}.pkl")
    
    with open(output_path, "wb") as f:
        pickle.dump({
            "all_token_embeddings": all_token_embeddings,
            "all_edge_index": all_edge_index,
            "y": y
        }, f)
    
    print(f"Saved to {output_path}")
    print(f"  - Embeddings: {len(all_token_embeddings)} graphs")
    print(f"  - Labels shape: {y.shape}")

if __name__ == '__main__':
    # Đường dẫn đến dữ liệu của bạn
    BASE_DIR = r"."
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    # Định nghĩa các file
    files = [
        ("hc3_train", os.path.join(DATA_DIR, "hc3_train.jsonl")),
        ("hc3_val", os.path.join(DATA_DIR, "hc3_val.jsonl")),
        ("hc3_test", os.path.join(DATA_DIR, "hc3_test.jsonl"))
    ]
    
    for name, file_path in files:
        print(f"\n{'='*50}")
        print(f"Processing {name}...")
        print(f"File: {file_path}")
        
        if not os.path.exists(file_path):
            print(f" File not found: {file_path}")
            print("Please make sure the file exists in the data directory.")
            continue
        
        # Load data
        texts, labels = load_jsonl_data(file_path)
        print(f"Loaded {len(texts)} samples")
        print(f"Label distribution - Human (0): {labels.count(0)}, AI (1): {labels.count(1)}")
        
        # Build graphs
        all_token_embeddings, all_edge_index, y = build_graph_from_texts(texts, labels)
        
        # Save
        save_pkl(name, all_token_embeddings, all_edge_index, y)
    
    print("\n Done! Graph data has been saved to 'graph_data/' directory")