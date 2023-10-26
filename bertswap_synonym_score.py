"""
BertSwapScore.py

Author: Nick Abegg

Description: This program takes 100 articles from Turingbench Dataset (50 human and 50 GPT3)
             It then genreates 10 probable words for each sentence and generates 10 new articles
             using these new words. The semantic similarit of all the articles are calculated in comparison
             to the orignal article. All of the data is then put into .csv
"""

from detector import OpenaiDetector
bearer_token = ''                   # enter bearer token for open AI

od = OpenaiDetector(bearer_token)

from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, DistilBertForMaskedLM

import torch
#import torch.nn.functional as F

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

from nltk.corpus import stopwords
nltk.download('stopwords')

#import numpy as np

import pandas as pd

#import re
import copy

import spacy

import matplotlib.pyplot as mp
import random

import csv

import time

#################################
#################################

nlp = spacy.load('en_core_web_sm')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForMaskedLM.from_pretrained("distilbert-base-uncased")

#score = score_bert("This is a sentence that I just wrote.")

punctuation_list = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

# gets a tokenizer from nltk to parse out each respective sentence
sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# setsup stop words using NLTK's stopword list
stops = set(stopwords.words('english'))

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


new_text_list = []
old_text_list = []

data_list = []

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
filename = 'BertSwap_UID_Semantic_Results.csv'  # Replace with your CSV file name
column_name1 = 'uid_score1'  # Replace with the desired column name
column_name2 = 'uid_score3'

column_data1 = get_column_data(filename, column_name1)
column_data2 = get_column_data(filename, column_name2)

# %%
#########################
#########################

"""Extracts synonyms for a given phrase using WordNet.

Args:
    phrase (str): The phrase for which synonyms are to be extracted.

Returns:
    list: A list of synonyms for the given phrase.
"""

def synonym_extractor(phrase):
    from nltk.corpus import wordnet
    synonyms = []

    for syn in wordnet.synsets(phrase):
        for l in syn.lemmas():
          if l.name() != phrase:
            synonyms.append(l.name())
    return synonyms

""" Gets the semantic similarity score of the sentences compared to the original string

Args:
    original (str): The original article that all the articles are compared to
    sentences (str): The altered articles

Returns:
    the cosine similarity score of all the articles given in a list
"""

def get_semantic_similarity (original, sentences):
    semantic_model = SentenceTransformer('bert-base-nli-mean-tokens')
    
    sentences_embeddings = semantic_model.encode(sentences)
    original_embeddings = semantic_model.encode(original)
    return(cosine_similarity(
        [original_embeddings],
        sentences_embeddings
    ))


"""
Takes a phrase/sentence/word and returns the part of speech of each word
Given: word(str) A phrase/sentence/word
Returns: a list of strings with the part of speech of each word/character in the given string
"""
def get_pos(word):
    parsed = nlp(word)
    pos_list = []
    for token in parsed:
        pos_list.append(token.pos_)
    return pos_list


'''
The bert swap function gets a sentence with a masked token inserted into the sentence. It then calculates the n most probable words according to BERT
and then creates 10 sentences with those words. Creates dataframes and then .csv for showing the probabilties for the words
Given: masked(str): the selected sentence with a masking token ([MASK]) inserted into the sentence
Returns: most_prob_sentences (list): A list of strings with the 10 most probable words
'''
def bert_swap(masked):
    
    # get tokens for masked sentence
    inputs = tokenizer(masked, return_tensors="pt")
    
    # calculates the probability of the sentence
    with torch.no_grad():
        logits = model(**inputs).logits
    
    # retrieve index of [MASK]
    mask_token_index = int((inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0])
    
    # get 10 most probable tokens
    most_prob_tokens = torch.argsort(logits[0,mask_token_index])[-10:]
    
    # decode those tokens
    most_prob_words = tokenizer.decode(most_prob_tokens)
    
    most_prob_words = most_prob_words.split()
    prob_words_list.append(most_prob_words)
    
    most_prob_sentences = []
    
    # insert the 10 most probable tokens into 10 seperate sentences
    for i in range(len(most_prob_words)):
        new_sentence = masked.replace('[MASK]', most_prob_words[i])
        most_prob_sentences.append(new_sentence)
    
    return most_prob_sentences


    
'''
Parses through all the sentences of an article from the dataset
Given: text(str) a full article from the dataset
       num_article(int) the nmber of the article in the dataset
Returns: the most probable sentences and the original sentence
'''
def parse_sentences(text, num_article):
    
    # parses out sentences
    sentences = sentence_tokenizer.tokenize(text['Generation'])
    #label = text['label']
    #result_df = pd.DataFrame()

    old_sentences = copy.copy(sentences)    # makes a copy for comparison output
    
    prob_sentences = []
    
    # for every sentence, select a suitable word, then change that word with a suitable synonym
    for i in range(len(sentences)):
        
        # gets each word/character in each sentence
        words = word_tokenize(sentences[i])
        
        # gets the pos for every word of the sentence
        pos = get_pos(' '.join(words))
        
        target_index = -1
        
        if len(words) >= 3:
            # picks a word in the middle
            word = words[int(len(words)/2)]
            
            target_index = int(len(words)/2)
        
        if not(target_index == -1):
        
            # chekcs to see if the word is a suitable replacement or not
            while ((word in stops) or (word in punctuation_list) or not(synonym_extractor(word)) or len(word) < 2 or not(word.isalpha()) or pos[target_index] == 'PROPN') and target_index < len(words)-1:
                target_index += 1
                word = words[target_index]
            
            
            if not(target_index == len(words)):    
                # calls the method that changes the target word to the proper synonym
                replaced = copy.copy(words)
                replaced[target_index] = 'REPLACED'
                
                masked_sentence = copy.copy(words)
                masked_sentence[target_index] = '[MASK]'
                
                replaced = ' '.join(replaced)
                masked_sentence = ' '.join(masked_sentence)
                
                
                prob_sentences.append((bert_swap(masked_sentence)))
        else:
            prob_words_list.append(None)
            prob_sentences.append(None)
    return prob_sentences, old_sentences
   

            
def create_DF(result, original, article, num_article):
    
    new_article_list = []
    
    # for every sentence, check if 10 sentences were produced,
    for i in range(len(result)):
        new_article = []
        if result[i] is not None:
            for num in range (len(result[i])):
                new_article.append(result[i][num])
            # add original sentences if not enough
            while not (len(new_article) == 10):
                new_article.append(original[i])
            
        else:
            # if none, then just use the original sentence
            new_article = [original[i]] * 10
            
        new_article_list.append(new_article)
        if prob_words_list[i] is not None:
            while not(len(prob_words_list[i]) == 10):
                prob_words_list[i].append(None)
    
    # create 10 new articles
    for y in range(10):
        sentences = ''
        if y == 0:
            article_list.append(' '.join(original))
        for i in range(len(result)):
            sentences = sentences + " " + new_article_list[i][y]
        article_list.append(sentences)
        
    # sorts which probable words were used in each article
    all_prob_words = []
    for y in range(10):
        if y == 0:
            all_prob_words.append([None] * len(result))
        temp_list = []
        for i in range(len(result)):
            if prob_words_list[i] is not None:
                temp_list.append(prob_words_list[i][y])
            else:
                temp_list.append(None)
        all_prob_words.append(temp_list)
    
    index = len(article_list) - 11
    score_list.extend(list(get_semantic_similarity(' '.join(original), article_list[index:len(article_list)])[0]))
    temp_uid_list = [num + random.randint(1, 10) for num in score_list[index:len(article_list)]]
    prob_word_list.extend(all_prob_words)
    
    return

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

# %%
# def plot(index, max_index, article, num_article, temp_uid_list):
    
#     # Example usage
#     filename = 'BertSwap_UID_Semantic_Results.csv'  # Replace with your CSV file name
#     column_name1 = 'uid_score1'  # Replace with the desired column name
#     column_name2 = 'uid_score3'

#     column_data1 = get_column_data(filename, column_name1)
#     column_data2 = get_column_data(filename, column_name2)
    
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
        if score_list[score_index] >= .98:
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
        if score_list[score_index] >= .98:
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
####################
### MAIN ###########
####################
article_score_df_list = []
article_list = []
score_list = []
temp_uid_list = []
original_list = []
prob_word_list = []

selected_article_list = []
original_article_list = []

# used for only human data
# for i in range(len(human)):
#     start_parse_sentence(human[i])

# used for gpt3 only data
# for i in range(len(gpt3)):
#     start_parse_sentence(gpt3[i])

# used for combine data of gpt3 and human
for num_article in range(len(human_gpt3)):
    
    prob_words_list = []
    
    # gets the most probable sentences
    result, original = parse_sentences(human_gpt3[num_article], num_article)
    
    original_list.append(original)
    
    create_DF(result, original, human_gpt3[num_article], num_article)
    
    candidate_select(num_article, human_gpt3[num_article])
    

# %%
article_df = pd.DataFrame()

article_df['Original Articles'] = original_article_list
article_df['Selected Articles'] = selected_article_list
article_df['True Labes'] = ['Human'] * 100 + ['GPT-3'] * 100

article_df.to_csv('BERTSwap_Selected_Articles.csv', index = False)


# %%
results_label = []
results_prob = []
for num_article in range(len(article_list)):
    
    detect_result = od.detect(article_list[num_article])
    if (detect_result == 'Check prompt, Length of sentence it should be more than 1,000 characters'):
        results_label.append(None)
        results_prob.append(None)
    else:
        results_label.append(detect_result['Class'])
        results_prob.append(detect_result['AI-Generated Probability'])
        
    if num_article % 5 == 0 and not(num_article == 0):
        time.sleep(10)
    
# %%
# first 0-449 indices are human, rest are gpt-3
result_df = pd.DataFrame()

result_df['Articles'] = article_list

result_df['Semantic Scores'] = score_list

result_df['Selected Words'] = prob_word_list

result_df['Detector Label'] = results_label

result_df['Detector Probability'] = results_prob

result_df.to_csv('BertSwap_Results.csv', index = False)
        