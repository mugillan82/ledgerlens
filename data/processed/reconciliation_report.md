# AI Finance Reconciliation Report
Generated: 2026-08-31 10:21:55

## Pipeline Summary
- **Total Bank Transactions**: 488
- **Total Gateway Settlement Records**: 490

- **Exact Matches**: 338 (69.26%)
- **ML Matches**: 136 (27.87%) (Avg Confidence: 98.01%)
- **Unmatched Bank Transactions (Exceptions)**: 14 (2.87%)
- **Unmatched Gateway Transactions (Exceptions)**: 16 (3.27%)

- **Overall Match Rate (Bank)**: 97.13%

## Business Impact Report
- **Total Transaction Value Processed**: ₹7,32,71,127.65
- **Total Value Auto-Reconciled**: ₹7,09,16,792.66 (96.79% of total value)
- **Total Value Flagged as Exceptions (Requires Audit)**: ₹23,54,334.99
- **Estimated Audit Time Saved**: 23.70 hours (based on ~3 min per auto-matched record)

## Validation Against Ground Truth
- **Reconciliation Accuracy**: 97.54%
- **Precision**: 98.31%
- **Recall**: 98.73%
- **Confusion Matrix**:
  - True Positives (Correct Matches): 466
  - True Negatives (Correct Unmatched): 10
  - False Positives (Incorrect Matches): 8
  - False Negatives (Missed Matches): 6

### Mismatches / Errors:
- Bank Txn `TXN-00234`: Pipeline matched to `ORD-00478` but ground truth is `ORD-00091` (Mismatched (GT: ORD-00091, Pipe: ORD-00478))
- Bank Txn `TXN-00383`: Pipeline matched to `ORD-00405` but ground truth is `ORD-00478` (Mismatched (GT: ORD-00478, Pipe: ORD-00405))
- Bank Txn `TXN-00341`: Pipeline matched to `ORD-00145` but ground truth is `NO_MATCH` (Incorrectly matched unmatched bank entry)
- Bank Txn `TXN-00239`: Pipeline matched to `ORD-00178` but ground truth is `NO_MATCH` (Incorrectly matched unmatched bank entry)
- Bank Txn `TXN-00388`: Pipeline matched to `ORD-00176` but ground truth is `NO_MATCH` (Incorrectly matched unmatched bank entry)
- Bank Txn `TXN-00021`: Pipeline matched to `ORD-00040` but ground truth is `NO_MATCH` (Incorrectly matched unmatched bank entry)
- Bank Txn `TXN-00011`: Pipeline matched to `ORD-00480` but ground truth is `NO_MATCH` (Incorrectly matched unmatched bank entry)
- Bank Txn `TXN-00170`: Pipeline matched to `ORD-00023` but ground truth is `NO_MATCH` (Incorrectly matched unmatched bank entry)
