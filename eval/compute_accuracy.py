# -*- coding: utf-8 -*-
"""
compute_accuracy.py

This script calculates the accuracy rate from a model log file.
It counts total samples and the number of correct predictions ("acc:True").
"""

# Path to the log file
file_path = "./Qwen2.5-3B_V-Session_MATH500.log"

# Initialize counters
total = 0
true_count = 0

# Read the log file line by line
with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        # Identify lines containing accuracy information
        if "acc:" in line:
            total += 1
            if "acc:True" in line:
                true_count += 1

# Calculate accuracy rate
if total > 0:
    acc_rate = true_count / total * 100
else:
    acc_rate = 0.0

# Print the results
print(f"Total samples: {total}")
print(f"Number of acc:True: {true_count}")
print(f"Accuracy: {acc_rate:.2f}%")
