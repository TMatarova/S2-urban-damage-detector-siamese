import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve, 
                             average_precision_score)
import os
import json
from pathlib import Path

# --------------------------
# Config - Match your Model.py paths
# --------------------------
# Get script location and project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Use the same paths as Model.py
OUT_DIR = PROJECT_ROOT / "Outputs" / "outputs"  # Same as your Model.py

# Load your model's results
npz_path = OUT_DIR / "model_outputs.npz"
metrics_path = OUT_DIR / "metrics.json"

print(f"Loading results from: {OUT_DIR}")

# --------------------------
# Load Data
# --------------------------
# Load predictions and labels
data = np.load(npz_path)
labels = data["labels"]
probs = data["probs"]

# Load existing metrics from JSON
with open(metrics_path, 'r') as f:
    metrics = json.load(f)

# --------------------------
# Calculate Metrics
# --------------------------
# ROC curve and AUC
fpr, tpr, roc_thresholds = roc_curve(labels, probs)
auc = roc_auc_score(labels, probs)

# Precision-Recall curve and Average Precision
precision_curve, recall_curve, pr_thresholds = precision_recall_curve(labels, probs)
ap = average_precision_score(labels, probs)

# --------------------------
# Create Combined Plot
# --------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curve
ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'Siamese CNN (AUC = {auc:.3f})')
ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curve - Siamese CNN with ResNet-101', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Precision-Recall Curve
ax2.plot(recall_curve, precision_curve, 'r-', linewidth=2, label=f'Siamese CNN (AP = {ap:.3f})')
ax2.axhline(y=labels.mean(), color='k', linestyle='--', linewidth=1, 
            label=f'Baseline (Rate = {labels.mean():.3f})')
ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Curve - Siamese CNN with ResNet-101', fontsize=14, fontweight='bold')
ax2.legend(loc='lower left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save_path = OUT_DIR / "siamese_roc_pr_curves.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()

# --------------------------
# Individual ROC Curve
# --------------------------
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2.5, label=f'Siamese CNN\n(AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7)
plt.fill_between(fpr, tpr, alpha=0.2, color='blue')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Sentinel-2 Damage Detection\nSiamese CNN with ResNet-101', fontsize=14)
plt.legend(loc='lower right', fontsize=11)
plt.grid(True, alpha=0.3)
save_path = OUT_DIR / "siamese_roc_curve.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()

# --------------------------
# Individual Precision-Recall Curve
# --------------------------
plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, 'r-', linewidth=2.5, label=f'Siamese CNN\n(AP = {ap:.3f})')
plt.fill_between(recall_curve, precision_curve, alpha=0.2, color='red')
plt.axhline(y=labels.mean(), color='gray', linestyle='--', linewidth=1.5, alpha=0.7, 
            label=f'Baseline\n(Rate = {labels.mean():.2%})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve - Sentinel-2 Damage Detection\nSiamese CNN with ResNet-101', fontsize=14)
plt.legend(loc='lower left', fontsize=11)
plt.grid(True, alpha=0.3)
save_path = OUT_DIR / "siamese_pr_curve.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.show()

# --------------------------
# Performance Summary
# --------------------------
performance_summary = {
    "Model": "Siamese CNN (ResNet-101)",
    "Patch Size": "16x16 (160m x 160m)",
    "AUC-ROC": round(auc, 4),
    "Average Precision": round(ap, 4),
    "Accuracy": round(metrics.get('accuracy', 0), 4),
    "Damage Precision": round(metrics.get('Damage', {}).get('precision', 0), 4),
    "Damage Recall": round(metrics.get('Damage', {}).get('recall', 0), 4),
    "Damage F1": round(metrics.get('Damage', {}).get('f1-score', 0), 4),
    "No-Damage Precision": round(metrics.get('No-Damage', {}).get('precision', 0), 4),
    "No-Damage Recall": round(metrics.get('No-Damage', {}).get('recall', 0), 4),
    "No-Damage F1": round(metrics.get('No-Damage', {}).get('f1-score', 0), 4)
}

# Save performance summary
save_path = OUT_DIR / "siamese_performance_summary.json"
with open(save_path, 'w') as f:
    json.dump(performance_summary, f, indent=4)

# Print summary
print("\n" + "="*60)
print("SIAMESE CNN PERFORMANCE SUMMARY")
print("="*60)
print(f"AUC-ROC: {performance_summary['AUC-ROC']:.4f}")
print(f"Average Precision: {performance_summary['Average Precision']:.4f}")
print(f"Accuracy: {performance_summary['Accuracy']:.2%}")
print("\nDamage Detection:")
print(f"  Precision: {performance_summary['Damage Precision']:.2%}")
print(f"  Recall: {performance_summary['Damage Recall']:.2%}")
print(f"  F1-Score: {performance_summary['Damage F1']:.4f}")
print("="*60)
print(f"\nAll results saved to: {OUT_DIR}")