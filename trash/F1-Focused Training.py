# 1. Find the Optimal Decision Threshold
precisions, recalls, thresholds = precision_recall_curve(y, oof_probs)
# Calculate F1 score for every possible threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]

print(f"Optimal Decision Threshold: {best_thresh:.4f}")
print(f"Best Validation F1 Score: {f1_scores[best_idx]:.4f}")

# 2. Generate Final Classification Report
y_pred_optimized = (oof_probs >= best_thresh).astype(int)

print("\n--- Final Validation Report ---")
print(classification_report(y, y_pred_optimized))

# 3. Plot Confusion Matrix
cm = confusion_matrix(y, y_pred_optimized)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not TDE', 'TDE'], yticklabels=['Not TDE', 'TDE'])
plt.title(f'Confusion Matrix (Threshold={best_thresh:.3f})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()