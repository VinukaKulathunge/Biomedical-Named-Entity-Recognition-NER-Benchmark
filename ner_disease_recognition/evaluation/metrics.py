from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

def calculate_ner_metrics(predictions: list, targets: list) -> dict:
    """
    Calculates token-level micro-precision, micro-recall, and micro-F1 scores
    for disease entity recognition, strictly ignoring 'O' tags in the scoring process.
    
    Args:
        predictions (list of list of str): Predicted BIO tags for each sentence in the dataset.
        targets (list of list of str): Ground-truth BIO tags for each sentence in the dataset.
        
    Returns:
        dict: A dictionary containing:
            - "precision": micro precision score (float).
            - "recall": micro recall score (float).
            - "f1": micro F1 score (float).
            - "report": detailed classification report (str).
    """
    # Flatten both predictions and targets
    flat_preds = [tag for seq in predictions for tag in seq]
    flat_targets = [tag for seq in targets for tag in seq]
    
    # We strictly evaluate on disease tags, ignoring the "O" class
    eval_labels = ["B-Disease", "I-Disease"]
    
    # Compute metrics using scikit-learn
    # Using labels=eval_labels restricts micro-averaging strictly to disease classes
    precision = precision_score(
        flat_targets, 
        flat_preds, 
        average="micro", 
        labels=eval_labels, 
        zero_division=0
    )
    
    recall = recall_score(
        flat_targets, 
        flat_preds, 
        average="micro", 
        labels=eval_labels, 
        zero_division=0
    )
    
    f1 = f1_score(
        flat_targets, 
        flat_preds, 
        average="micro", 
        labels=eval_labels, 
        zero_division=0
    )
    
    # Generate detailed token-level classification report
    report = classification_report(
        flat_targets, 
        flat_preds, 
        labels=eval_labels, 
        zero_division=0
    )
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report
    }
