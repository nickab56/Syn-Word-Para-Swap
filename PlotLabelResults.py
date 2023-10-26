import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

# %%
def get_column_data(filename, column_name):
    data_list = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        if column_name not in reader.fieldnames:
            print(f"Column '{column_name}' not found in the CSV file.")
            return data_list
        
        for row in reader:
            if not( row[column_name] == ''):
                data_list.append(int(row[column_name]))
            else:
                data_list.append(None)
    
    return data_list

def handle_none_values(data):
    return [0 if x is None else x for x in data]

# Example usage
filename = 'ParaphraseSwap_Labels.csv'  # Replace with your CSV file name
#filename = 'BERTSwap_Labels.csv'  # Replace with your CSV file name

column_name1 = 'GPTZero_Original'  # Replace with the desired column name
column_name2 = 'GPTZero_Selected'
column_name3 = 'DetectGPT_Original'
column_name4 = 'DetectGPT_Selected'

column_data1 = get_column_data(filename, column_name1)
column_data2 = get_column_data(filename, column_name2)
column_data3 = get_column_data(filename, column_name3)
column_data4 = get_column_data(filename, column_name4)


# Group the data for plotting
data = [column_data1, column_data2, column_data3, column_data4]
labels = ["GPT Zero Original", 'GPT Zero Obfuscated', 'DetectGPT Original', 'DetectGPT Obfuscated']

# Count occurrences of 0, 1, and None for each column
zeros = [col.count(0) for col in data]
ones = [col.count(1) for col in data]
nones = [col.count(None) for col in data]

# Set the width of the bars
bar_width = 0.2

# Position of bars on the x-axis
r1 = range(len(zeros))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Create a larger figure to accommodate the labels
plt.figure(figsize=(10, 6))

# Create the grouped bar chart
plt.bar(r1, zeros, color='red', width=bar_width, edgecolor='black', label='Machine')
plt.bar(r2, ones, color='green', width=bar_width, edgecolor='black', label='Human')
plt.bar(r3, nones, color='blue', width=bar_width, edgecolor='black', label='None')

# Customize the plot
plt.xlabel('Authorship Detectors')
plt.ylabel('Number of Labels')
plt.title('UID Word Swap Obfuscated Article Lables vs. Originals')
plt.xticks([r + 1.5 * bar_width for r in range(len(zeros))], labels)  # Increased spacing between bars

# Move the legend outside of the graph
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), title='Values')

# Show the plot
plt.tight_layout()
plt.show()

# %%

column_data1 = handle_none_values(column_data1)
column_data2 = handle_none_values(column_data2)
column_data3 = handle_none_values(column_data3)
column_data4 = handle_none_values(column_data4)

true_labels = [1] * 100 + [0] * 100
true_labels1 = column_data1
predicted_labels1 = column_data2
true_labels2 = column_data3
predicted_labels2 = column_data4


f1_score1 = f1_score(true_labels, predicted_labels1, average='micro')#gptZero
f1_score2 = f1_score(true_labels, predicted_labels2, average='micro')#detectGPT

# Print the F1 scores
print(f"F1 Score for GPT Zero: {f1_score1}")#gptZero
print(f"F1 Score for DetectGPT: {f1_score2}")#detectGPT

conf_matrix1 = confusion_matrix(true_labels, predicted_labels1) #gptZero
conf_matrix2 = confusion_matrix(true_labels, predicted_labels2) #detectGpt


