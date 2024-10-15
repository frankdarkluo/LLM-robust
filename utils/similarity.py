from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = 'false'
import itertools
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

def cal_sim(word1, word2):

    # Encode the cities into embeddings
    embedding1 = model.encode(word1, convert_to_tensor=True)
    embedding2 = model.encode(word2, convert_to_tensor=True)

    # Compute the cosine similarity between the two embeddings
    similarity = util.pytorch_cos_sim(embedding1, embedding2)

    # Print the similarity score
    # print(f"Semantic similarity between {word1} and {word2}: {similarity.item():.4f}")
    return similarity.item()

# Function to find the most semantically similar word for each word in a list
def find_most_similar_for_each_word(word_list):
    most_similar_dict = {}

    # Iterate over each word in the list
    for word in word_list:
        similarities = {}
        
        # Compare the current word with every other word in the list
        for other_word in word_list:
            if word != other_word:  # Avoid comparing the word with itself
                similarity = cal_sim(word, other_word)
                if similarity is not None:  # Only include valid similarities
                    similarities[other_word] = similarity

        if similarities:  # Ensure there are valid similarity scores
            # Find the word with the highest similarity score
            most_similar_word = max(similarities, key=similarities.get)
            most_similar_dict[word] = most_similar_word

    return most_similar_dict

def find_most_similar(word, word_list):
    similarities = {}
    
    # Iterate over the list of words and compute similarity with 'word'
    for other_word in word_list:
        similarity = cal_sim(word, other_word)
        if similarity is not None:  # Only include valid similarities
            similarities[other_word] = similarity
    

    # Find the word with the highest similarity score
    most_similar_word = max(similarities, key=similarities.get)
    print(f"The word most semantically similar to '{word}' is '{most_similar_word}' with a similarity score of {similarities[most_similar_word]:.4f}")
    return most_similar_word, similarities[most_similar_word]

def find_min_similar(word, word_list):
    similarities = {}
    
    # Iterate over the list of words and compute similarity with 'word'
    for other_word in word_list:
        similarity = cal_sim(word, other_word)
        if similarity is not None:  # Only include valid similarities
            similarities[other_word] = similarity
    

    # Find the word with the highest similarity score
    min_similar_word = min(similarities, key=similarities.get)
    print(f"The word least semantically similar to '{word}' is '{min_similar_word}' with a similarity score of {similarities[min_similar_word]:.4f}")
    return min_similar_word, similarities[min_similar_word]

def cluster(cities):
    # Encode all the cities using the pre-trained model
    # prompt='The capital of Brody Raion is [Y] .'
    # cities =[prompt.replace('[Y]', city) for city in cities]
    city_embeddings = model.encode(cities)

    # Number of clusters (adjust as needed)
    num_clusters = 3

    # Apply KMeans clustering on the embeddings
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(city_embeddings)

    # Get the cluster assignments
    city_clusters = kmeans.labels_

    # Print the clustered cities
    for i in range(num_clusters):
        cluster = [cities[j] for j in range(len(cities)) if city_clusters[j] == i]
        print(f"Cluster {i+1}: {cluster}")

if __name__ == "__main__":
    word1="beijing"
    word2="baku"
    # most_similar_dict=find_most_similar_for_each_word(cities)
    similarity_dict = {}
    sentences = [
    'The native language of Francis Ponge is Russian .',
    'The native language of Francis Ponge is Chinese .',
    'The native language of Francis Ponge is Telugu .',
    'The native language of Francis Ponge is Dutch .',
    'The native language of Francis Ponge is English .']

    for (i, j) in itertools.combinations(range(len(sentences)), 2):
        # Construct the sentence pair key
        sentence_pair = f"{sentences[i]} - {sentences[j]}"
        reversed_pair = f"{sentences[j]} - {sentences[i]}"
        # Calculate the similarity using precomputed embeddings
        similarity_score = cal_sim(sentences[i], sentences[j])
        
        # Store the similarity score in the dictionary
        similarity_dict[sentence_pair] = similarity_score
        similarity_dict[reversed_pair] = similarity_score

    # Print the similarity dictionary
    for pair, score in similarity_dict.items():
        print(f"{pair} : {score}")
        # cal_sim(word2, cities)
        # cluster(cities)
