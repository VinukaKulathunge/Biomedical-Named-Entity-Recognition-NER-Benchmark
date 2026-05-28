import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time

from tokenizer_custom import WhitespaceTokenizer, NLTKTokenizer, BPETokenizer
from embeddings import Word2VecEmbedding, GloVeEmbedding, FastTextEmbedding, ELMoEmbedding, BertEmbedding
from models import BiLSTM_CRF, BiLSTM_Softmax
from evaluation import calculate_ner_metrics
from utils import BC5CDRDataLoader

# Mapping tags to indices
TAG2IDX = {"O": 0, "B-Disease": 1, "I-Disease": 2}
IDX2TAG = {0: "O", 1: "B-Disease", 2: "I-Disease"}

class NERDataset(Dataset):
    """
    Standard PyTorch dataset that stores tokens and tags, and exposes helper methods
    for mapping text to vocabulary indices based on the active embedding.
    """
    def __init__(self, aligned_data: list, embedding_wrapper):
        self.data = aligned_data
        self.embedding_wrapper = embedding_wrapper
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        tokens, tags = self.data[idx]
        
        # Determine representation based on embedding type
        # BertEmbedding handles string inputs directly. Static/ELMo use indices.
        if isinstance(self.embedding_wrapper, BertEmbedding):
            input_repr = tokens # list of strings
        else:
            # Map strings to indices using the embedding's built-in vocabulary mapper
            input_repr = self.embedding_wrapper.text_to_indices(tokens)
            
        tag_indices = [TAG2IDX[t] for t in tags]
        return input_repr, tag_indices, tokens, tags


def pad_collate_fn(batch, embedding_wrapper, is_crf: bool):
    """
    Custom collation function to pad variable-length sequences within a batch.
    
    Args:
        batch: List of tuples (input_repr, tag_indices, raw_tokens, raw_tags)
        embedding_wrapper: The current embedding module.
        is_crf (bool): Whether the active model uses a CRF classification layer.
        
    Returns:
        tuple: Padded tensors, boolean mask, raw tokens, and true tags.
    """
    # 1. Sort the batch by length in descending order (useful for RNN packing if needed)
    batch.sort(key=lambda x: len(x[3]), reverse=True)
    
    input_reprs, tag_indices_list, raw_tokens_list, raw_tags_list = zip(*batch)
    
    lengths = [len(tags) for tags in raw_tags_list]
    max_len = max(lengths)
    
    batch_size = len(batch)
    
    # 2. Pad input tokens/indices
    is_bert = isinstance(embedding_wrapper, BertEmbedding)
    
    if is_bert:
        # BERT takes list of lists of strings, padding is handled internally inside BertEmbedding
        padded_inputs = list(input_reprs)
    else:
        # Static and ELMo take padded long tensors
        padded_inputs = torch.zeros(batch_size, max_len, dtype=torch.long)
        for i, seq in enumerate(input_reprs):
            padded_inputs[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            
    # 3. Pad tags
    # For CRF, we can pad with a valid tag (e.g. 0/O) since the mask blocks its influence.
    # For Softmax, we pad with -1 so CrossEntropyLoss's ignore_index ignores it.
    pad_tag_idx = 0 if is_crf else -1
    padded_tags = torch.full((batch_size, max_len), pad_tag_idx, dtype=torch.long)
    for i, seq in enumerate(tag_indices_list):
        padded_tags[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        
    # 4. Generate masking tensor (True/1 for real tokens, False/0 for pad tokens)
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
        
    return padded_inputs, padded_tags, mask, raw_tokens_list, raw_tags_list


class NERPipeline:
    """
    NER Pipeline supporting modular assembly, training, evaluation, and logging
    of any tokenizer, embedding, and model combination.
    """
    def __init__(
        self, 
        tokenizer_name: str, 
        embedding_name: str, 
        classifier_name: str,
        hidden_dim: int = 128,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 0.001
    ):
        self.tokenizer_name = tokenizer_name
        self.embedding_name = embedding_name
        self.classifier_name = classifier_name
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.embedding = None
        self.model = None
        
    def setup_pipeline(self, train_raw: list):
        """
        Initializes the tokenizers, trains/downloads the embeddings,
        and constructs the BiLSTM model.
        """
        # 1. Initialize Tokenizer
        if self.tokenizer_name == "Whitespace":
            self.tokenizer = WhitespaceTokenizer()
        elif self.tokenizer_name == "NLTK":
            self.tokenizer = NLTKTokenizer()
        elif self.tokenizer_name == "BPE":
            self.tokenizer = BPETokenizer()
        else:
            raise ValueError(f"Unknown tokenizer: {self.tokenizer_name}")
            
        # Extract training sentences tokenized by target tokenizer to fit embeddings
        print(f"  [Setup] Aligning training dataset using '{self.tokenizer_name}' tokenizer...")
        train_aligned = []
        train_sentences_tokens = []
        for sent in train_raw:
            new_tokens, new_tags = BC5CDRDataLoader.align_tokens_and_tags(sent, self.tokenizer)
            train_aligned.append((new_tokens, new_tags))
            train_sentences_tokens.append(new_tokens)
            
        # 2. Initialize Embeddings
        print(f"  [Setup] Setting up '{self.embedding_name}' embedding...")
        if self.embedding_name == "Word2Vec":
            self.embedding = Word2VecEmbedding(sentences=train_sentences_tokens, dim=100)
        elif self.embedding_name == "GloVe":
            self.embedding = GloVeEmbedding(sentences=train_sentences_tokens, dim=50)
        elif self.embedding_name == "FastText":
            self.embedding = FastTextEmbedding(sentences=train_sentences_tokens, dim=100)
        elif self.embedding_name == "ELMo":
            self.embedding = ELMoEmbedding(sentences=train_sentences_tokens, word_dim=128, lstm_hidden=128)
        elif self.embedding_name == "BERT":
            self.embedding = BertEmbedding(model_name="bert-base-uncased")
        else:
            raise ValueError(f"Unknown embedding: {self.embedding_name}")
            
        self.embedding.to(self.device)
        embedding_dim = self.embedding.get_dim()
        
        # 3. Initialize Model (Sequence model + Classification layer)
        is_crf = (self.classifier_name == "CRF")
        num_tags = len(TAG2IDX)
        
        print(f"  [Setup] Initializing BiLSTM-{self.classifier_name} model (Dim: {embedding_dim} -> {self.hidden_dim * 2} -> {num_tags})...")
        if is_crf:
            self.model = BiLSTM_CRF(
                embedding_dim=embedding_dim, 
                hidden_dim=self.hidden_dim, 
                num_tags=num_tags
            )
        else:
            self.model = BiLSTM_Softmax(
                embedding_dim=embedding_dim, 
                hidden_dim=self.hidden_dim, 
                num_tags=num_tags
            )
            
        self.model.to(self.device)
        return train_aligned

    def run(self, train_raw: list, dev_raw: list, test_raw: list) -> dict:
        """
        Executes setup, training, validation, and testing of the pipeline.
        
        Returns:
            dict: Evaluation results containing F1-score, Precision, Recall, and Report.
        """
        start_time = time.time()
        
        # 1. Setup components
        train_aligned = self.setup_pipeline(train_raw)
        
        # Align dev and test sets
        dev_aligned = []
        for sent in dev_raw:
            new_tokens, new_tags = BC5CDRDataLoader.align_tokens_and_tags(sent, self.tokenizer)
            dev_aligned.append((new_tokens, new_tags))
            
        test_aligned = []
        for sent in test_raw:
            new_tokens, new_tags = BC5CDRDataLoader.align_tokens_and_tags(sent, self.tokenizer)
            test_aligned.append((new_tokens, new_tags))
            
        # 2. Build PyTorch Datasets
        train_dataset = NERDataset(train_aligned, self.embedding)
        dev_dataset = NERDataset(dev_aligned, self.embedding)
        test_dataset = NERDataset(test_aligned, self.embedding)
        
        is_crf = (self.classifier_name == "CRF")
        
        # 3. Create DataLoaders with custom collation
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=lambda b: pad_collate_fn(b, self.embedding, is_crf)
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            collate_fn=lambda b: pad_collate_fn(b, self.embedding, is_crf)
        )
        
        # 4. Train model
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        
        print(f"  [Pipeline] Training '{self.tokenizer_name} + {self.embedding_name} + BiLSTM-{self.classifier_name}' on {self.device}...")
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            
            # Progress bar for training
            pbar = tqdm(train_loader, desc=f"  Epoch {epoch}/{self.epochs}", leave=False)
            for padded_inputs, padded_tags, mask, _, _ in pbar:
                optimizer.zero_grad()
                
                # Dynamic embedding generation
                if isinstance(self.embedding, BertEmbedding):
                    # BERT takes list of lists of token strings directly
                    # input is dynamic list: padded_inputs
                    embeddings = self.embedding(padded_inputs)
                else:
                    # Static/ELMo embeddings take integer index tensors
                    padded_inputs = padded_inputs.to(self.device)
                    embeddings = self.embedding(padded_inputs)
                    
                padded_tags = padded_tags.to(self.device)
                mask = mask.to(self.device)
                
                # Forward pass
                loss = self.model(embeddings, padded_tags, mask)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                
            avg_loss = total_loss / len(train_loader)
            print(f"    Epoch {epoch}/{self.epochs} - Average Loss: {avg_loss:.4f}")
            
        # 5. Evaluate on test set
        print("  [Pipeline] Evaluating model on the BC5CDR test split...")
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for padded_inputs, padded_tags, mask, raw_tokens_list, raw_tags_list in test_loader:
                # Dynamic embedding extraction
                if isinstance(self.embedding, BertEmbedding):
                    embeddings = self.embedding(padded_inputs)
                else:
                    padded_inputs = padded_inputs.to(self.device)
                    embeddings = self.embedding(padded_inputs)
                    
                mask = mask.to(self.device)
                
                # Decode tag indices using active sequence decoder
                # decoded_ids: list of lists of tag indices matching true sentence boundaries
                decoded_ids = self.model.decode(embeddings, mask)
                
                # Map indices back to string tags and filter out pad markers
                for i in range(len(raw_tokens_list)):
                    pred_tags = [IDX2TAG[idx] for idx in decoded_ids[i]]
                    true_tags = raw_tags_list[i]
                    
                    # Truncate predicted tags to true length if necessary
                    pred_tags = pred_tags[:len(true_tags)]
                    # Pad predictions if decoded returned fewer tokens
                    if len(pred_tags) < len(true_tags):
                        pred_tags.extend(["O"] * (len(true_tags) - len(pred_tags)))
                        
                    all_preds.append(pred_tags)
                    all_targets.append(true_tags)
                    
        # Calculate F1 metrics ignoring O
        metrics = calculate_ner_metrics(all_preds, all_targets)
        metrics["runtime"] = time.time() - start_time
        
        print(f"  [Pipeline Done] micro F1-score: {metrics['f1']:.4f} (Runtime: {metrics['runtime']:.2f}s)")
        
        return metrics
