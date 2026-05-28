import argparse
import sys
import os

# Ensure the parent directory is in the path to allow clean imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import BC5CDRDataLoader
from pipeline import NERPipeline

def run_single(args):
    print("=" * 70)
    print("                BC5CDR DISEASE NER - SINGLE PIPELINE RUN             ")
    print("=" * 70)
    print(f"  Tokenizer:  {args.tokenizer}")
    print(f"  Embedding:  {args.embedding}")
    print(f"  Classifier: {args.classifier}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Hidden Dim: {args.hidden_dim}")
    print(f"  Learning Rate: {args.lr}")
    print("=" * 70)
    
    # 1. Load data
    loader = BC5CDRDataLoader(data_dir="data")
    train_raw, dev_raw, test_raw = loader.load_data()
    
    if args.fast:
        print("\n[Fast Mode Enabled] Restricting to a small data subset (100 train, 30 test) for speed.")
        train_raw = train_raw[:100]
        dev_raw = dev_raw[:30]
        test_raw = test_raw[:30]
        
    # 2. Construct and execute the selected pipeline
    pipeline = NERPipeline(
        tokenizer_name=args.tokenizer,
        embedding_name=args.embedding,
        classifier_name=args.classifier,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    metrics = pipeline.run(train_raw, dev_raw, test_raw)
    
    print("\n" + "=" * 70)
    print("                             FINAL METRICS                          ")
    print("=" * 70)
    print(f"  Micro Precision: {metrics['precision']:.4f}")
    print(f"  Micro Recall:    {metrics['recall']:.4f}")
    print(f"  Micro F1-Score:  **{metrics['f1']:.4f}**")
    print(f"  Total Runtime:   {metrics['runtime']:.1f}s")
    print("=" * 70)
    print("\nToken-level detailed classification report:\n")
    print(metrics["report"])

def main():
    parser = argparse.ArgumentParser(
        description="Modular BC5CDR Biomedical Named Entity Recognition Suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--run-all", 
        action="store_true", 
        help="Runs the complete 30-configuration grid benchmark and generates a Markdown report."
    )
    
    parser.add_argument(
        "--tokenizer", 
        type=str, 
        default="BPE", 
        choices=["Whitespace", "NLTK", "BPE"], 
        help="Tokenizer type for word/subword segmentation."
    )
    
    parser.add_argument(
        "--embedding", 
        type=str, 
        default="BERT", 
        choices=["Word2Vec", "GloVe", "FastText", "ELMo", "BERT"], 
        help="Representation embeddings layer."
    )
    
    parser.add_argument(
        "--classifier", 
        type=str, 
        default="CRF", 
        choices=["CRF", "Softmax"], 
        help="Classification layer type."
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=3, 
        help="Number of training epochs."
    )
    
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16, 
        help="Mini-batch size."
    )
    
    parser.add_argument(
        "--hidden_dim", 
        type=int, 
        default=64, 
        help="Hidden dimension size of the BiLSTM."
    )
    
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001, 
        help="Optimizer learning rate."
    )
    
    parser.add_argument(
        "--fast", 
        action="store_true", 
        default=True,
        help="Subsets data for fast evaluation."
    )
    
    parser.add_argument(
        "--full", 
        dest="fast", 
        action="store_false", 
        help="Disables subsetting to run on the entire corpus."
    )
    
    args = parser.parse_args()
    
    if args.run_all:
        # Import and run all experiments
        from experiments import run_all
        # Set FAST_MODE inside run_all to match CLI
        run_all.FAST_MODE = args.fast
        run_all.main()
    else:
        run_single(args)

if __name__ == "__main__":
    main()
