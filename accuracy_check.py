import pandas as pd

# Read the CSV file into a DataFrame
csv_file_path = "D:\Projects\RAG_pdf\Evaluation_dataset.csv"  # replace with the actual path to your CSV file
df = pd.read_csv(csv_file_path)

# Calculate the accuracy
df['Accuracy'] = df['Response'] == df['Answer_from_model']

# Save the updated DataFrame to a new CSV file
updated_csv_file_path = "final_dataset.csv"
df.to_csv(updated_csv_file_path, index=False)

print("Updated CSV file saved to:", updated_csv_file_path)
