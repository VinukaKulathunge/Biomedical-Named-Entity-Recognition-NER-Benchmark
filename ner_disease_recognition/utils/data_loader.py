import os
import urllib.request
import re
import random

class BC5CDRDataLoader:
    """
    Handles downloading, parsing, and aligning of the BC5CDR disease NER dataset.
    """
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.urls = {
            "train": "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC5CDR-disease-IOB/train.tsv",
            "dev": "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC5CDR-disease-IOB/devel.tsv",
            "test": "https://raw.githubusercontent.com/cambridgeltl/MTL-Bioinformatics-2016/master/data/BC5CDR-disease-IOB/test.tsv"
        }
        
    def load_data(self):
        """
        Loads the train, dev, and test sets. Downloads from GitHub if not present.
        If offline and files are missing, generates synthetic medical dataset.
        
        Returns:
            tuple: (train_sents, dev_sents, test_sents) where each is a list of
                   sentences: list of tuples (word, tag).
        """
        splits = ["train", "dev", "test"]
        data = {}
        
        for split in splits:
            file_path = self._get_file_path(split)
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"[Data Loader] Local data for '{split}' not found at {file_path}.")
                success = self._download_split(split, file_path)
                if not success:
                    print(f"[Data Loader] Failed to download '{split}'. Generating synthetic medical corpus...")
                    self._generate_synthetic_data()
                    
            # Parse the CoNLL style file
            data[split] = self._parse_conll_file(self._get_file_path(split))
            print(f"[Data Loader] Loaded {len(data[split])} sentences for '{split}'.")
            
        return data["train"], data["dev"], data["test"]
        
    def _get_file_path(self, split: str):
        # Allow both .txt and .tsv
        p_txt = os.path.join(self.data_dir, f"{split}.txt")
        p_tsv = os.path.join(self.data_dir, f"{split}.tsv")
        if os.path.exists(p_tsv) and not os.path.exists(p_txt):
            return p_tsv
        return p_txt
        
    def _download_split(self, split: str, file_path: str) -> bool:
        url = self.urls[split]
        print(f"  --> Downloading BC5CDR dataset split '{split}' from Cambridge LTL GitHub repository...")
        try:
            # Add user agent to avoid HTTP 403
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8')
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
            print(f"  --> Successfully downloaded '{split}' to {file_path}")
            return True
        except Exception as e:
            print(f"  [Download Error] Failed downloading from {url}: {e}")
            return False
            
    def _parse_conll_file(self, file_path: str):
        """
        Parses a token-per-line CoNLL format file.
        Sentences are separated by empty lines.
        """
        sentences = []
        current_sentence = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Empty line represents end of sentence
                if not line:
                    if len(current_sentence) > 0:
                        sentences.append(current_sentence)
                        current_sentence = []
                    continue
                    
                # Support both spaces and tabs
                parts = re.split(r'\s+', line)
                if len(parts) >= 2:
                    word, tag = parts[0], parts[-1]
                    # Map any non-disease tags (e.g. Chemical) to O if they leak
                    if tag not in ["B-Disease", "I-Disease", "O"]:
                        if "Disease" in tag:
                            pass # keep B-Disease, I-Disease
                        else:
                            tag = "O"
                    current_sentence.append((word, tag))
                    
            if len(current_sentence) > 0:
                sentences.append(current_sentence)
                
        return sentences

    def _generate_synthetic_data(self):
        """
        Generates realistic, synthetically created disease NER sentences in BIO format.
        Guarantees code executability without internet or original dataset files.
        """
        print("[Data Loader] Synthesizing medically realistic disease recognition dataset...")
        
        templates = [
            ("The patient was admitted with symptoms of B-Disease I-Disease .", ["acute", "myocardial", "infarction"]),
            ("No history of B-Disease I-Disease was reported by the patient .", ["coronary", "artery", "disease"]),
            ("We evaluated the subject for B-Disease I-Disease .", ["diabetes", "mellitus"]),
            ("She developed B-Disease I-Disease after taking the drug .", ["acute", "kidney", "injury"]),
            ("Laboratory results indicated B-Disease .", ["hepatitis"]),
            ("The physician diagnosed the child with B-Disease I-Disease .", ["otitis", "media"]),
            ("He has a familial history of B-Disease .", ["hypertension"]),
            ("We suspected B-Disease I-Disease in this patient .", ["pulmonary", "congestion"]),
            ("Biopsy confirmed the diagnosis of B-Disease .", ["neoplasm"]),
            ("Signs of B-Disease I-Disease were observed in the chest X-ray .", ["pleural", "effusion"]),
            ("The treatment for B-Disease was successful .", ["diabetes"]),
            ("She has been suffering from chronic B-Disease for five years .", ["arthritis"]),
            ("The patient presented with B-Disease I-Disease I-Disease .", ["end", "stage", "renal", "disease"]),
            ("Clinical findings were consistent with B-Disease I-Disease .", ["heart", "failure"]),
            ("Risk factors include B-Disease and obesity .", ["hypercholesterolemia"])
        ]
        
        for split in ["train", "dev", "test"]:
            file_path = os.path.join(self.data_dir, f"{split}.txt")
            sentences_count = 120 if split == "train" else (50 if split == "dev" else 50)
            
            with open(file_path, "w", encoding="utf-8") as f:
                for _ in range(sentences_count):
                    template, entities = random.choice(templates)
                    words = template.split()
                    
                    for w in words:
                        if w.startswith("B-Disease") and w.endswith("I-Disease"):
                            # Handle two word entity
                            f.write(f"{entities[0]}\tB-Disease\n")
                            f.write(f"{entities[1]}\tI-Disease\n")
                        elif w == "B-Disease" and len(entities) == 1:
                            f.write(f"{entities[0]}\tB-Disease\n")
                        elif w == "B-Disease" and len(entities) >= 2:
                            # Write entity tokens
                            f.write(f"{entities[0]}\tB-Disease\n")
                            for sub_ent in entities[1:]:
                                f.write(f"{sub_ent}\tI-Disease\n")
                        elif w == "B-Disease" and len(entities) == 0:
                            # Fallback if empty
                            f.write(f"disease\tB-Disease\n")
                        else:
                            f.write(f"{w}\tO\n")
                    f.write("\n") # Blank line between sentences

    @staticmethod
    def align_tokens_and_tags(original_sentence: list, tokenizer) -> tuple:
        """
        Aligns the original word tokens and their BIO tags with a new tokenizer's splits.
        Uses a robust character-level tracking algorithm.
        
        Args:
            original_sentence (list of tuple): List of (word, tag) from the dataset.
            tokenizer: The custom tokenizer wrapper (Whitespace, NLTK, or BPE).
            
        Returns:
            tuple: (new_tokens, new_tags) where:
                - new_tokens is a list of token strings from the tokenizer.
                - new_tags is a list of aligned tags.
        """
        orig_words = [item[0] for item in original_sentence]
        orig_tags = [item[1] for item in original_sentence]
        
        # 1. Reconstruct sentence and map character indices to original token index
        text = ""
        char_to_orig_idx = {}
        for i, word in enumerate(orig_words):
            start = len(text)
            text += word
            end = len(text)
            for c_idx in range(start, end):
                char_to_orig_idx[c_idx] = i
            # Add a space between tokens if not the last token
            if i < len(orig_words) - 1:
                text += " "
                
        # 2. Tokenize the reconstructed text
        new_tokens, spans = tokenizer.tokenize(text)
        
        # 3. Align tags based on overlapping spans
        new_tags = []
        # Keep track of which original tokens have already been tagged B-Disease
        # so subsequent sub-tokens get mapped to I-Disease (subword tagging logic)
        first_subtoken_assigned = set()
        
        for j, (start, end) in enumerate(spans):
            # Special case for empty spans (e.g. HuggingFace special tokens like [CLS], [SEP])
            if start == 0 and end == 0:
                new_tags.append("O")
                continue
                
            # Find the original token index overlapping with this sub-token start character
            # If the start character falls on a space, find the first valid character index inside the span
            orig_idx = None
            for idx in range(start, end):
                if idx in char_to_orig_idx:
                    orig_idx = char_to_orig_idx[idx]
                    break
                    
            if orig_idx is None:
                # If no overlap, map to O (e.g. extra punctuation or padding)
                new_tags.append("O")
                continue
                
            orig_tag = orig_tags[orig_idx]
            
            if orig_tag == "O":
                new_tags.append("O")
            elif orig_tag == "B-Disease":
                if orig_idx not in first_subtoken_assigned:
                    new_tags.append("B-Disease")
                    first_subtoken_assigned.add(orig_idx)
                else:
                    new_tags.append("I-Disease")
            elif orig_tag == "I-Disease":
                new_tags.append("I-Disease")
            else:
                new_tags.append("O")
                
        return new_tokens, new_tags
