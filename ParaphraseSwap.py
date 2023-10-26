"""
ParphraseSwap.py

Author: Nick Abegg

Paraphrasing code from: https://towardsdatascience.com/high-quality-sentence-paraphraser-using-transformers-in-nlp-c33f4482856f

"""

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
phrase_model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
phrase_tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")


import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
phrase_model = phrase_model
#phrase_model = torch.nn.DataParallel(phrase_model)

model_path = "detector-large.pt"
detector_model = torch.load(model_path)  # Load the model

model_dict = torch.load('detector-Large.pt', map_location='cuda')  # You can specify 'cuda' if using GPU

semantic_model = SentenceTransformer('bert-base-nli-mean-tokens')

import copy

import matplotlib.pyplot as mp

import csv

import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
# gets a tokenizer from nltk to parse out each respective sentence
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# loads the dataset
from datasets import load_dataset

dataset = load_dataset("turingbench/TuringBench")


human = []
# filters through the dataset for the first 50 human labeled text
i = 0
while len(human) < 50:
    if dataset['train'][i]['label'] == 'human':
        human.append(dataset['train'][i])
    i += 1
    
    
gpt3 = []
# filters through the dataset for the first 50 gpt3 generaetd text
i = 0
while len(gpt3) < 50:
    if dataset['train'][i]['label'] == 'gpt3':
        gpt3.append(dataset['train'][i])
    i += 1


# combines both datasets
human_gpt3 = human + gpt3

# %%
def get_column_data(filename, column_name):
    data_list = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        if column_name not in reader.fieldnames:
            print(f"Column '{column_name}' not found in the CSV file.")
            return data_list
        
        for row in reader:
            data_list.append(float(row[column_name]))
    
    return data_list

# Example usage
filename = 'ParaphraseSwapUID.csv'  # Replace with your CSV file name
column_name1 = 'uid_score1'  # Replace with the desired column name
column_name2 = 'uid_score3'

column_data1 = get_column_data(filename, column_name1)
column_data2 = get_column_data(filename, column_name2)

# %%

######################################################

""" Gets the semantic similarity score of the sentences compared to the original string

Args:
    original (str): The original article that all the articles are compared to
    sentences (str): The altered articles

Returns:
    the cosine similarity score of all the articles given in a list
"""

def get_semantic_similarity (original, sentences):
    sentences_embeddings = semantic_model.encode(sentences)
    original_embeddings = semantic_model.encode(original)
    return(cosine_similarity(
        [original_embeddings],
        sentences_embeddings
    ))

def ParaSwap(sentence):
    new_articles = []
    # Diverse Beam search
    context =  sentence
    text = context + " </s>"
    encoding = phrase_tokenizer.encode_plus(text, padding=True, return_tensors="pt")
    input_ids,attention_mask  = encoding["input_ids"], encoding["attention_mask"]
    phrase_model.eval()
    diverse_beam_outputs = phrase_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=10,
        num_beam_groups=10,
        num_return_sequences=10,
        diversity_penalty=0.70,
        max_length=500,  # Increase the max_length value as needed
        )
    print ("\n\n")
    print ("Original: ",context)
    for beam_output in diverse_beam_outputs:
        new_articles.append(phrase_tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True).strip('paraphrasedoutput:'))
    return new_articles

def parse_sentences(text):
    new_sentences_list = []
    
    # parses out sentences
    sentences = sentence_tokenizer.tokenize(text)

    old_sentences = copy.copy(sentences)    # makes a copy for comparison output
    
    # for every sentence, select a suitable word, then change that word with a suitable synonym
    for i in range(len(sentences)):
        if(len(sentences[i].split()) >= 8):
            new_sentences_list.append(ParaSwap(sentences[i]))
        else:
            new_sentences_list.append(list([sentences[i]]*10))
    return new_sentences_list

def make_articles(sentences_list, original_article):
     # create 10 new articles
     for y in range(10):
         new = []
         if y == 0:
             article_list.append(original_article)
         for i in range(len(sentences_list)):
             new.append(sentences_list[i][y])
         article_list.append(' '.join(new))

     index = len(article_list) - 11  
     score_list.extend(list(get_semantic_similarity(original_article, article_list[index:len(article_list)])[0]))


# %%
# def plot(article):
#     index = len(article_list) - 11
#     max_index = len(article_list)
    
#     pass
#     # Set colors for each data point
#     colors = ['red'] + ['blue'] * (len(column_data1[index:max_index]) - 1)  # First point in red, rest in blue
    
#     mp.scatter(score_list[index:max_index], column_data1[index:max_index], c=colors)
#     mp.xlabel("Semantic Score")
#     mp.ylabel('UID Score 1')
#     mp.title(f"Semantic VS UID Article {article['label']} {num_article}")
    
#     # Create a legend
#     original_patch = mp.Line2D([], [], marker='o', markersize=8, color='red', linestyle='', label='Original Article')
#     new_patch = mp.Line2D([], [], marker='o', markersize=8, color='blue', linestyle='', label='New Articles')
#     mp.legend(handles=[original_patch, new_patch], bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     mp.show()
    
#     # Clear the current figure
#     mp.clf()
    
#     mp.scatter(score_list[index:max_index], column_data2[index:max_index], c=colors)
#     mp.xlabel("Semantic Score")
#     mp.ylabel('UID Score 3')
#     mp.title(f"Semantic VS UID Article {article['label']} {num_article}")
    
#     # Create a legend
#     original_patch = mp.Line2D([], [], marker='o', markersize=8, color='red', linestyle='', label='Original Article')
#     new_patch = mp.Line2D([], [], marker='o', markersize=8, color='blue', linestyle='', label='New Articles')
#     mp.legend(handles=[original_patch, new_patch], bbox_to_anchor=(1.05, 1), loc='upper left')
    
#     mp.show()

def candidate_select(num_article, article):
    index = num_article * 11
    max_index = index + 11
    
    UID1_Difference_list = []
    UID2_Difference_list = []
    
    original_article = article_list[index]
    alternate_list = article_list[index+1:max_index]
    
    original_UID1 = column_data1[index]
    alternate_UID1 = column_data1[index+1:max_index]
    
    original_UID2 = column_data2[index]
    alternate_UID2 = column_data2[index+1:max_index]
    
    for i in range(len(alternate_UID1)):
        UID1_Difference_list.append(abs(original_UID1-alternate_UID1[i]))
            
    for i in range(len(alternate_UID2)):
        UID2_Difference_list.append(abs(original_UID2-alternate_UID2[i]))
        
    sorted_UID1 = sorted(UID1_Difference_list)
    sorted_UID2 = sorted(UID2_Difference_list)
    
    for i in reversed(sorted_UID1):
        score_index = UID1_Difference_list.index(i) + 1
        if score_list[score_index] >= .85:
            selected_article_list.append(alternate_list[score_index])
            break
    mp.clf()
    
    # Set colors for each data point
    colors = ['red'] + ['blue'] * (len(column_data1[index:max_index]) - 1)  # First point in red, rest in blue
    colors[score_index] = 'purple'
    
    mp.scatter(score_list[index:max_index], column_data1[index:max_index], c=colors)
    mp.xlabel("Semantic Score")
    mp.ylabel('UID Score 1')
    mp.title(f"Semantic VS UID Article {article['label']} {num_article}")
    
    # Create a legend
    original_patch = mp.Line2D([], [], marker='o', markersize=8, color='red', linestyle='', label='Original Article')
    new_patch = mp.Line2D([], [], marker='o', markersize=8, color='blue', linestyle='', label='New Articles')
    select_patch = mp.Line2D([], [], marker='o', markersize=8, color='purple', linestyle='', label='Selected Article')
    mp.legend(handles=[original_patch, new_patch, select_patch], bbox_to_anchor=(1.05, 1), loc='upper left')
    
    mp.show()
    
    for i in reversed(sorted_UID2):
        score_index = UID2_Difference_list.index(i) + 1
        if score_list[score_index] >= .85:
            selected_article_list.append(alternate_list[score_index])
            break
        
    mp.clf()
     
    # Set colors for each data point
    colors = ['red'] + ['blue'] * (len(column_data1[index:max_index]) - 1)  # First point in red, rest in blue
    colors[score_index] = 'purple'
     
    mp.scatter(score_list[index:max_index], column_data2[index:max_index], c=colors)
    mp.xlabel("Semantic Score")
    mp.ylabel('UID Score 3')
    mp.title(f"Semantic VS UID Article {article['label']} {num_article}")
     
    # Create a legend
    original_patch = mp.Line2D([], [], marker='o', markersize=8, color='red', linestyle='', label='Original Article')
    new_patch = mp.Line2D([], [], marker='o', markersize=8, color='blue', linestyle='', label='New Articles')
    select_patch = mp.Line2D([], [], marker='o', markersize=8, color='purple', linestyle='', label='Selected Article')
    mp.legend(handles=[original_patch, new_patch, select_patch], bbox_to_anchor=(1.05, 1), loc='upper left')
     
    mp.show()
    
    original_article_list.append(original_article)
    original_article_list.append(original_article)
    
    
    
    
# %%
article_list = []
score_list = []
selected_article_list = []
original_article_list = []

# used for combine data of gpt3 and human
for num_article in range(len(human_gpt3)):
    article_sentences = parse_sentences(human_gpt3[num_article]['Generation'])
    make_articles(article_sentences, human_gpt3[num_article]['Generation'])
    candidate_select(num_article, human_gpt3[num_article])



# %%
article_df = pd.DataFrame()

article_df['Original Articles'] = original_article_list
article_df['Selected Articles'] = selected_article_list
article_df['True Labes'] = ['Human'] * 100 + ['GPT-3'] * 100

article_df.to_csv('ParaphraseSwap_Selected_Articles.csv', index = False)

# %%
#plot()

# first 0-449 indices are human, rest are gpt-3
result_df = pd.DataFrame()

result_df['Articles'] = article_list

result_df['Semantic Scores'] = score_list

result_df.to_csv('ParaphraseSwap.csv', index = False)

pass