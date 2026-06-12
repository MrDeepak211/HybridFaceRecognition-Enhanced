import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import os
import time
import json

def evaluate_model(model, X_test, y_test, class_names=None, save_path='results/'):
    """
    Evaluates the model and plots/saves the confusion matrix.
    """
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1': f1_score(y_test, y_pred, average='macro')
    }
    
    print("\n" + "="*30)
    print("Model Evaluation Results")
    print("="*30)
    for k, v in metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")
    print("-" * 30)
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto')
    plt.title('Confusion Matrix - Hybrid Model')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    return metrics


def compute_comprehensive_metrics(model, X_test, y_test, class_names=None, save_path='results/'):
    """
    Compute comprehensive evaluation metrics including:
    - Accuracy, Precision, Recall, F1-Score
    - Per-class performance
    - Confusion Matrix
    - ROC-AUC curves
    - Inference time
    - Error analysis
    """
    os.makedirs(save_path, exist_ok=True)
    
    print("\nComputing comprehensive metrics...")
    
    # Predictions
    start_time = time.time()
    y_pred_probs = model.predict(X_test, verbose=0)
    inference_time = (time.time() - start_time) / len(y_test) * 1000  # ms per image
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Overall metrics
    overall_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
        'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        'recall_weighted': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
        'inference_time_ms': float(inference_time)
    }
    
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:          {overall_metrics['accuracy']:.4f} ({overall_metrics['accuracy']*100:.2f}%)")
    print(f"Precision (macro): {overall_metrics['precision_macro']:.4f}")
    print(f"Recall (macro):    {overall_metrics['recall_macro']:.4f}")
    print(f"F1-Score (macro):  {overall_metrics['f1_macro']:.4f}")
    print(f"Inference Time:    {overall_metrics['inference_time_ms']:.2f} ms/image")
    
    # Per-class metrics
    per_class_report = classification_report(
        y_test, y_pred, 
        target_names=class_names if class_names else None,
        output_dict=True,
        zero_division=0
    )
    
    print(f"\n{'='*60}")
    print(f"PER-CLASS PERFORMANCE")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto')
    plt.title('Confusion Matrix - Hybrid Face Recognition Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=150)
    plt.close()
    
    print(f"Confusion matrix saved to {save_path}/confusion_matrix.png")
    
    # ROC-AUC (One-vs-Rest)
    num_classes = y_pred_probs.shape[1]
    if num_classes == 2:
        # Binary classification
        roc_auc = float(roc_auc_score(y_test, y_pred_probs[:, 1]))
        overall_metrics['roc_auc'] = roc_auc
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_probs[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(save_path, 'roc_curve.png'), dpi=150)
        plt.close()
    else:
        # Multi-class: One-vs-Rest
        y_test_bin = label_binarize(y_test, classes=range(num_classes))
        
        # Compute ROC-AUC for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Micro-average ROC-AUC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_probs.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        overall_metrics['roc_auc_micro'] = float(roc_auc["micro"])
        
        # Macro-average ROC-AUC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= num_classes
        roc_auc["macro"] = auc(all_fpr, mean_tpr)
        overall_metrics['roc_auc_macro'] = float(roc_auc["macro"])
        
        # Plot ROC curves
        plt.figure(figsize=(10, 8))
        plt.plot(fpr["micro"], tpr["micro"],
                label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.4f})',
                color='deeppink', linestyle=':', linewidth=4)
        
        plt.plot(fpr["macro"], tpr["macro"],
                label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.4f})',
                color='navy', linestyle=':', linewidth=4)
        
        # Plot individual classes (limit to first 10 for clarity)
        colors = plt.cm.tab20(np.linspace(0, 1, num_classes))
        for i in range(min(num_classes, 10)):
            plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                    label=f'Class {i} (AUC = {roc_auc[i]:.4f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-Class ROC Curves')
        plt.legend(loc='lower right', fontsize=8)
        plt.savefig(os.path.join(save_path, 'roc_curves.png'), dpi=150)
        plt.close()
    
    print(f"ROC-AUC (micro): {overall_metrics.get('roc_auc_micro', 'N/A')}")
    print(f"ROC-AUC (macro): {overall_metrics.get('roc_auc_macro', 'N/A')}")
    
    # Error Analysis
    errors = y_test != y_pred
    error_rate = np.mean(errors)
    
    error_analysis = {
        'total_samples': int(len(y_test)),
        'correct_predictions': int(np.sum(~errors)),
        'incorrect_predictions': int(np.sum(errors)),
        'error_rate': float(error_rate),
        'most_confused_pairs': []
    }
    
    # Find most confused pairs
    if class_names:
        confused_pairs = []
        for i in range(len(cm)):
            for j in range(len(cm)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append({
                        'true_class': class_names[i],
                        'predicted_class': class_names[j],
                        'count': int(cm[i, j])
                    })
        
        # Sort by count and get top 5
        confused_pairs.sort(key=lambda x: x['count'], reverse=True)
        error_analysis['most_confused_pairs'] = confused_pairs[:5]
    
    print(f"\n{'='*60}")
    print(f"ERROR ANALYSIS")
    print(f"{'='*60}")
    print(f"Total Samples: {error_analysis['total_samples']}")
    print(f"Correct: {error_analysis['correct_predictions']}")
    print(f"Incorrect: {error_analysis['incorrect_predictions']}")
    print(f"Error Rate: {error_analysis['error_rate']*100:.2f}%")
    
    if error_analysis['most_confused_pairs']:
        print(f"\nMost Confused Pairs:")
        for pair in error_analysis['most_confused_pairs'][:3]:
            print(f"  {pair['true_class']} -> {pair['predicted_class']}: {pair['count']} times")
    
    # Accuracy by confidence threshold
    confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    confidence_analysis = {}
    max_confidences = np.max(y_pred_probs, axis=1)
    
    for threshold in confidence_thresholds:
        mask = max_confidences >= threshold
        if np.sum(mask) > 0:
            acc_at_threshold = accuracy_score(y_test[mask], y_pred[mask])
            confidence_analysis[str(threshold)] = {
                'accuracy': float(acc_at_threshold),
                'num_samples': int(np.sum(mask)),
                'percentage': float(np.mean(mask) * 100)
            }
    
    overall_metrics['confidence_analysis'] = confidence_analysis
    
    # Save all metrics to JSON
    all_metrics = {
        'overall': overall_metrics,
        'per_class': per_class_report,
        'error_analysis': error_analysis,
        'confusion_matrix': cm.tolist()
    }
    
    with open(os.path.join(save_path, 'comprehensive_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    print(f"\nAll metrics saved to {save_path}/comprehensive_metrics.json")
    
    return overall_metrics


def plot_training_history(history_dict, save_path='results/'):
    """
    Plot training history for all stages
    """
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Accuracy plot
    axes[0, 0].set_title('Training Accuracy')
    for stage, hist in history_dict.items():
        if 'accuracy' in hist:
            axes[0, 0].plot(hist['accuracy'], label=f'{stage} Train')
        if 'val_accuracy' in hist:
            axes[0, 0].plot(hist['val_accuracy'], label=f'{stage} Val', linestyle='--')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss plot
    axes[0, 1].set_title('Training Loss')
    for stage, hist in history_dict.items():
        if 'loss' in hist:
            axes[0, 1].plot(hist['loss'], label=f'{stage} Train')
        if 'val_loss' in hist:
            axes[0, 1].plot(hist['val_loss'], label=f'{stage} Val', linestyle='--')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'training_history.png'), dpi=150)
    plt.close()
    
    print(f"Training history plot saved to {save_path}/training_history.png")


def compare_models(metrics_dict, save_path='results/'):
    """
    Compare multiple model configurations
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Extract accuracies
    models = list(metrics_dict.keys())
    accuracies = [metrics_dict[m]['accuracy'] for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['#667eea', '#764ba2', '#4facfe', '#f093fb'])
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{acc*100:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.ylim(0.9, 1.0)
    plt.ylabel('Accuracy')
    plt.title('Model Comparison - Accuracy')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'model_comparison.png'), dpi=150)
    plt.close()
    
    print(f"Model comparison plot saved to {save_path}/model_comparison.png")
