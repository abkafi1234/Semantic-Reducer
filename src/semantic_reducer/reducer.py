import torch
import numpy as np
import faiss
import pickle
import os
import re
from tqdm import tqdm
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModel

class SemanticReducer:
    def __init__(self, model_name='bert-base-multilingual-cased', cache_dir='./cache'):
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Use Fast tokenizer for word_ids mapping
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval() # Ensure evaluation mode
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.word_counts = Counter()
        # Explicitly set dtype to float32 for FAISS compatibility
        self.word_embeddings_sum = defaultdict(lambda: np.zeros(self.model.config.hidden_size, dtype=np.float32))
        self.vocab_embeddings = None
        self.vocab_list = []
        self.index = None
        self.reduction_map = {} # O(1) inference dictionary

    def process_corpus_contextually(self, sentences, batch_size=16):
        """
        Processes entire sentences to extract sense-averaged contextual embeddings.
        Subword aggregation correctly implemented using word_ids.
        """
        print("Extracting contextual embeddings from corpus...")
        
        # FIX 1: Use regex to cleanly separate punctuation from words so they don't get glued together
        split_sentences = [re.findall(r'\w+|[^\w\s]', str(sent)) for sent in sentences]
        
        for i in tqdm(range(0, len(split_sentences), batch_size), desc="Processing Batches"):
            batch = split_sentences[i:i+batch_size]
            
            # Tokenize batch with word alignment (returns a BatchEncoding object)
            inputs = self.tokenizer(batch, is_split_into_words=True, return_tensors='pt', 
                                    padding=True, truncation=True, max_length=512)
            
            # Move tensors to device FOR THE MODEL, but do NOT overwrite 'inputs'
            model_inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**model_inputs)
            
            # Use last_hidden_state
            hidden_states = outputs.last_hidden_state.cpu().numpy()
            
            # Aggregate subwords into whole words
            for batch_idx in range(len(batch)):
                word_ids = inputs.word_ids(batch_index=batch_idx) 
                original_words = batch[batch_idx]
                
                # Group subword embeddings by word_id
                word_embeddings_temp = defaultdict(list)
                for seq_idx, word_idx in enumerate(word_ids):
                    if word_idx is not None: # Ignore [CLS], [SEP], [PAD]
                        word_embeddings_temp[word_idx].append(hidden_states[batch_idx, seq_idx, :])
                
                # Average subwords to get the word embedding, add to global corpus sum
                for word_idx, embeddings in word_embeddings_temp.items():
                    if word_idx < len(original_words):
                        word = original_words[word_idx]
                        avg_word_emb = np.mean(embeddings, axis=0)
                        
                        self.word_embeddings_sum[word] += avg_word_emb
                        self.word_counts[word] += 1

    def finalize_embeddings(self):
        """Averages the accumulated contextual embeddings to create static representations."""
        print("Finalizing sense-averaged embeddings...")
        self.vocab_list = list(self.word_counts.keys())
        self.vocab_embeddings = np.zeros((len(self.vocab_list), self.model.config.hidden_size), dtype=np.float32)
        
        for idx, word in enumerate(self.vocab_list):
            self.vocab_embeddings[idx] = self.word_embeddings_sum[word] / self.word_counts[word]
            
        # L2 Normalize for FAISS Inner Product (Cosine Similarity)
        faiss.normalize_L2(self.vocab_embeddings)

    def build_index(self):
        """Builds the FAISS index. Explicitly uses Inner Product (Cosine)."""
        dim = self.vocab_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim) 
        self.index.add(self.vocab_embeddings)
        print(f"Built FAISS index with {self.index.ntotal} vectors of dimension {dim}.")

    def build_reduction_map(self, threshold=0.9, top_k=10):
        """
        Creates the global mapping dictionary.
        """
        print(f"Building semantic reduction map (threshold={threshold})...")
        self.reduction_map = {}
        
        # Search all words against the index at once for efficiency
        distances, indices = self.index.search(self.vocab_embeddings, top_k)
        
        for i, word in enumerate(tqdm(self.vocab_list, desc="Mapping Vocabulary")):
            candidates = []
            for j in range(top_k):
                dist = distances[i][j]
                neighbor_idx = indices[i][j]
                
                # FIX 2: Check for -1 to prevent FAISS from wrapping around the list when vocab < top_k
                if neighbor_idx != -1 and dist >= threshold:
                    neighbor_word = self.vocab_list[neighbor_idx]
                    candidates.append(neighbor_word)
            
            if candidates:
                # Sort candidates by corpus frequency (descending)
                candidates.sort(key=lambda w: (self.word_counts[w], -len(w)), reverse=True)
                self.reduction_map[word] = candidates[0]
            else:
                self.reduction_map[word] = word

    def reduce_text(self, text):
        """O(1) inference for new text."""
        # FIX 3: Apply the same regex tokenization during inference so words match the dictionary
        tokens = re.findall(r'\w+|[^\w\s]', text)
        return ' '.join([self.reduction_map.get(t, t) for t in tokens])

    def save_system(self, prefix="semred"):
        """Saves the minimal artifacts needed for inference."""
        file_path = os.path.join(self.cache_dir, f"{prefix}_map.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(self.reduction_map, f)
        print(f"System saved successfully to {file_path}")

    def load_system(self, prefix="semred"):
        """Loads a previously saved mapping dictionary for fast inference."""
        file_path = os.path.join(self.cache_dir, f"{prefix}_map.pkl")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find a saved system at {file_path}")
            
        with open(file_path, 'rb') as f:
            self.reduction_map = pickle.load(f)
        print(f"System loaded successfully from {file_path}")
