---

CB-ST EASE 0.0677 0.0 0.0666 0.0687 True
CB-ST ItemKNN 0.0567 0.0 0.0556 0.0578 True
CB-ST Pop 0.024 0.0 0.023 0.0251 True
EASE ItemKNN -0.011 0.0 -0.0121 -0.0099 True
EASE Pop -0.0436 0.0 -0.0447 -0.0425 True
ItemKNN Pop -0.0326 0.0 -0.0337 -0.0315 True

---

--- Part 3: Interpretation for Claim 'CB-ST significantly underperforms' ---
✅ Comparison 'CB-ST' vs. 'EASE': SIGNIFICANT (p=0)
✅ Comparison 'CB-ST' vs. 'ItemKNN': SIGNIFICANT (p=0)
✅ Comparison 'CB-ST' vs. 'Pop': SIGNIFICANT (p=0)

🏆 FINAL CONCLUSION: The claim is STATISTICALLY SUBSTANTIATED for this condition.
CB-ST performs significantly worse than all other compared algorithms.
(base) ulyssemaes@Ulysses-MacBook-Pro-9 result-statistical-tests % cleart
zsh: command not found: cleart
(base) ulyssemaes@Ulysses-MacBook-Pro-9 result-statistical-tests % clear
(base) ulyssemaes@Ulysses-MacBook-Pro-9 result-statistical-tests % python index.py

Running analysis for dataset: adressa, K: 10
Mean NDCG for EASE on adressa @ 10: 0.0760
Mean NDCG for ItemKNN on adressa @ 10: 0.0663
Mean NDCG for Pop on adressa @ 10: 0.0488
Mean NDCG for CB-ST on adressa @ 10: 0.0019
================================================================================
🔬 STATISTICAL ANALYSIS FOR: adressa @ 10
================================================================================

--- Part 1: One-Way ANOVA ---
F-statistic: 6526.6489
P-value: 0
✅ Result: The p-value is less than 0.05. We reject the null hypothesis.
Conclusion: There is a statistically significant difference among the algorithms.

--- Part 2: Tukey's HSD Post-Hoc Test (Pairwise Comparisons) ---
Multiple Comparison of Means - Tukey HSD, FWER=0.05
=====================================================
group1 group2 meandiff p-adj lower upper reject

---

CB-ST EASE 0.074 0.0 0.0725 0.0755 True
CB-ST ItemKNN 0.0644 0.0 0.0629 0.0658 True
CB-ST Pop 0.0469 0.0 0.0454 0.0483 True
EASE ItemKNN -0.0097 0.0 -0.0111 -0.0082 True
EASE Pop -0.0272 0.0 -0.0286 -0.0257 True
ItemKNN Pop -0.0175 0.0 -0.019 -0.016 True

---

--- Part 3: Interpretation for Claim 'CB-ST significantly underperforms' ---
✅ Comparison 'CB-ST' vs. 'EASE': CB-ST significantly WORSE (p=0, meandiff=0.0740)
✅ Comparison 'CB-ST' vs. 'ItemKNN': CB-ST significantly WORSE (p=0, meandiff=0.0644)
✅ Comparison 'CB-ST' vs. 'Pop': CB-ST significantly WORSE (p=0, meandiff=0.0469)

🏆 FINAL CONCLUSION: The claim is STATISTICALLY SUBSTANTIATED for this condition.
CB-ST performs significantly worse than all other compared algorithms.

Running analysis for dataset: adressa, K: 20
Mean NDCG for EASE on adressa @ 20: 0.1007
Mean NDCG for ItemKNN on adressa @ 20: 0.0842
Mean NDCG for Pop on adressa @ 20: 0.0601
Mean NDCG for CB-ST on adressa @ 20: 0.0030
================================================================================
🔬 STATISTICAL ANALYSIS FOR: adressa @ 20
================================================================================

--- Part 1: One-Way ANOVA ---
F-statistic: 10426.0963
P-value: 0
✅ Result: The p-value is less than 0.05. We reject the null hypothesis.
Conclusion: There is a statistically significant difference among the algorithms.

--- Part 2: Tukey's HSD Post-Hoc Test (Pairwise Comparisons) ---
Multiple Comparison of Means - Tukey HSD, FWER=0.05
=====================================================
group1 group2 meandiff p-adj lower upper reject

---

CB-ST EASE 0.0977 0.0 0.0961 0.0992 True
CB-ST ItemKNN 0.0812 0.0 0.0796 0.0827 True
CB-ST Pop 0.0571 0.0 0.0556 0.0586 True
EASE ItemKNN -0.0165 0.0 -0.018 -0.015 True
EASE Pop -0.0406 0.0 -0.0421 -0.0391 True
ItemKNN Pop -0.0241 0.0 -0.0256 -0.0226 True

---

--- Part 3: Interpretation for Claim 'CB-ST significantly underperforms' ---
✅ Comparison 'CB-ST' vs. 'EASE': CB-ST significantly WORSE (p=0, meandiff=0.0977)
✅ Comparison 'CB-ST' vs. 'ItemKNN': CB-ST significantly WORSE (p=0, meandiff=0.0812)
✅ Comparison 'CB-ST' vs. 'Pop': CB-ST significantly WORSE (p=0, meandiff=0.0571)

🏆 FINAL CONCLUSION: The claim is STATISTICALLY SUBSTANTIATED for this condition.
CB-ST performs significantly worse than all other compared algorithms.

Running analysis for dataset: adressa, K: 50
Mean NDCG for EASE on adressa @ 50: 0.1383
Mean NDCG for ItemKNN on adressa @ 50: 0.1182
Mean NDCG for Pop on adressa @ 50: 0.0868
Mean NDCG for CB-ST on adressa @ 50: 0.0053
================================================================================
🔬 STATISTICAL ANALYSIS FOR: adressa @ 50
================================================================================

--- Part 1: One-Way ANOVA ---
F-statistic: 19591.3648
P-value: 0
✅ Result: The p-value is less than 0.05. We reject the null hypothesis.
Conclusion: There is a statistically significant difference among the algorithms.

--- Part 2: Tukey's HSD Post-Hoc Test (Pairwise Comparisons) ---
Multiple Comparison of Means - Tukey HSD, FWER=0.05
=====================================================
group1 group2 meandiff p-adj lower upper reject

---

CB-ST EASE 0.1329 0.0 0.1314 0.1344 True
CB-ST ItemKNN 0.1129 0.0 0.1114 0.1144 True
CB-ST Pop 0.0814 0.0 0.0799 0.0829 True
EASE ItemKNN -0.02 0.0 -0.0216 -0.0185 True
EASE Pop -0.0515 0.0 -0.053 -0.05 True
ItemKNN Pop -0.0315 0.0 -0.033 -0.03 True

---

--- Part 3: Interpretation for Claim 'CB-ST significantly underperforms' ---
✅ Comparison 'CB-ST' vs. 'EASE': CB-ST significantly WORSE (p=0, meandiff=0.1329)
✅ Comparison 'CB-ST' vs. 'ItemKNN': CB-ST significantly WORSE (p=0, meandiff=0.1129)
✅ Comparison 'CB-ST' vs. 'Pop': CB-ST significantly WORSE (p=0, meandiff=0.0814)

🏆 FINAL CONCLUSION: The claim is STATISTICALLY SUBSTANTIATED for this condition.
CB-ST performs significantly worse than all other compared algorithms.

Running analysis for dataset: ebnerd, K: 10
Mean NDCG for EASE on ebnerd @ 10: 0.0406
Mean NDCG for ItemKNN on ebnerd @ 10: 0.0311
Mean NDCG for Pop on ebnerd @ 10: 0.0052
Mean NDCG for CB-ST on ebnerd @ 10: 0.0016
================================================================================
🔬 STATISTICAL ANALYSIS FOR: ebnerd @ 10
================================================================================

--- Part 1: One-Way ANOVA ---
F-statistic: 5150.5127
P-value: 0
✅ Result: The p-value is less than 0.05. We reject the null hypothesis.
Conclusion: There is a statistically significant difference among the algorithms.

--- Part 2: Tukey's HSD Post-Hoc Test (Pairwise Comparisons) ---
Multiple Comparison of Means - Tukey HSD, FWER=0.05
=====================================================
group1 group2 meandiff p-adj lower upper reject

---

CB-ST EASE 0.039 0.0 0.038 0.04 True
CB-ST ItemKNN 0.0295 0.0 0.0285 0.0305 True
CB-ST Pop 0.0036 0.0 0.0026 0.0045 True
EASE ItemKNN -0.0095 0.0 -0.0105 -0.0085 True
EASE Pop -0.0354 0.0 -0.0364 -0.0345 True
ItemKNN Pop -0.0259 0.0 -0.0269 -0.025 True

---

--- Part 3: Interpretation for Claim 'CB-ST significantly underperforms' ---
✅ Comparison 'CB-ST' vs. 'EASE': CB-ST significantly WORSE (p=0, meandiff=0.0390)
✅ Comparison 'CB-ST' vs. 'ItemKNN': CB-ST significantly WORSE (p=0, meandiff=0.0295)
✅ Comparison 'CB-ST' vs. 'Pop': CB-ST significantly WORSE (p=0, meandiff=0.0036)

🏆 FINAL CONCLUSION: The claim is STATISTICALLY SUBSTANTIATED for this condition.
CB-ST performs significantly worse than all other compared algorithms.

Running analysis for dataset: ebnerd, K: 20
Mean NDCG for EASE on ebnerd @ 20: 0.0525
Mean NDCG for ItemKNN on ebnerd @ 20: 0.0418
Mean NDCG for Pop on ebnerd @ 20: 0.0082
Mean NDCG for CB-ST on ebnerd @ 20: 0.0023
================================================================================
🔬 STATISTICAL ANALYSIS FOR: ebnerd @ 20
================================================================================

--- Part 1: One-Way ANOVA ---
F-statistic: 7713.1064
P-value: 0
✅ Result: The p-value is less than 0.05. We reject the null hypothesis.
Conclusion: There is a statistically significant difference among the algorithms.

--- Part 2: Tukey's HSD Post-Hoc Test (Pairwise Comparisons) ---
Multiple Comparison of Means - Tukey HSD, FWER=0.05
=====================================================
group1 group2 meandiff p-adj lower upper reject

---

CB-ST EASE 0.0501 0.0 0.0491 0.0511 True
CB-ST ItemKNN 0.0395 0.0 0.0384 0.0405 True
CB-ST Pop 0.0059 0.0 0.0049 0.0069 True
EASE ItemKNN -0.0107 0.0 -0.0117 -0.0096 True
EASE Pop -0.0442 0.0 -0.0453 -0.0432 True
ItemKNN Pop -0.0336 0.0 -0.0346 -0.0326 True

---

--- Part 3: Interpretation for Claim 'CB-ST significantly underperforms' ---
✅ Comparison 'CB-ST' vs. 'EASE': CB-ST significantly WORSE (p=0, meandiff=0.0501)
✅ Comparison 'CB-ST' vs. 'ItemKNN': CB-ST significantly WORSE (p=0, meandiff=0.0395)
✅ Comparison 'CB-ST' vs. 'Pop': CB-ST significantly WORSE (p=0, meandiff=0.0059)

🏆 FINAL CONCLUSION: The claim is STATISTICALLY SUBSTANTIATED for this condition.
CB-ST performs significantly worse than all other compared algorithms.

Running analysis for dataset: ebnerd, K: 50
Mean NDCG for EASE on ebnerd @ 50: 0.0716
Mean NDCG for ItemKNN on ebnerd @ 50: 0.0606
Mean NDCG for Pop on ebnerd @ 50: 0.0280
Mean NDCG for CB-ST on ebnerd @ 50: 0.0039
================================================================================
🔬 STATISTICAL ANALYSIS FOR: ebnerd @ 50
================================================================================

--- Part 1: One-Way ANOVA ---
F-statistic: 10698.9952
P-value: 0
✅ Result: The p-value is less than 0.05. We reject the null hypothesis.
Conclusion: There is a statistically significant difference among the algorithms.

--- Part 2: Tukey's HSD Post-Hoc Test (Pairwise Comparisons) ---
Multiple Comparison of Means - Tukey HSD, FWER=0.05
=====================================================
group1 group2 meandiff p-adj lower upper reject

---

CB-ST EASE 0.0677 0.0 0.0666 0.0687 True
CB-ST ItemKNN 0.0567 0.0 0.0556 0.0578 True
CB-ST Pop 0.024 0.0 0.023 0.0251 True
EASE ItemKNN -0.011 0.0 -0.0121 -0.0099 True
EASE Pop -0.0436 0.0 -0.0447 -0.0425 True
ItemKNN Pop -0.0326 0.0 -0.0337 -0.0315 True

---

--- Part 3: Interpretation for Claim 'CB-ST significantly underperforms' ---
✅ Comparison 'CB-ST' vs. 'EASE': CB-ST significantly WORSE (p=0, meandiff=0.0677)
✅ Comparison 'CB-ST' vs. 'ItemKNN': CB-ST significantly WORSE (p=0, meandiff=0.0567)
✅ Comparison 'CB-ST' vs. 'Pop': CB-ST significantly WORSE (p=0, meandiff=0.0240)

🏆 FINAL CONCLUSION: The claim is STATISTICALLY SUBSTANTIATED for this condition.
CB-ST performs significantly worse than all other compared algorithms.
