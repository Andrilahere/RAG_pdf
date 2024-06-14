import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

updated_csv_file_path = r"D:\Projects\RAG_pdf\final_dataset.csv"
df = pd.read_csv(updated_csv_file_path)

accuracy = accuracy_score(df['Response'], df['Answer_from_model'])

true_labels = df['Response'] == df['Response']
predicted_labels = df['Answer_from_model'] == df['Response']


precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
f1 = f1_score(true_labels, predicted_labels)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
