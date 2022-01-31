import spacy
import json
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.corpus import words
from flask import Flask, request, render_template
import os
import json


app = Flask(__name__)

directory = os.listdir('E:/Aidetic/663_20170904095457/data')
# Note: This directory ('E:/Aidetic/663_20170904095457/data') is based on my directory name, you can change it according to yours
# Imp: The directory you put must contain atleast 20 json files.

list_of_articles = []
for i in directory[:20]:
    with open('E:/Aidetic/663_20170904095457/data/'+i, 'rb') as f:
        file = json.load(f)
        list_of_articles.append(file['text'])

class Extracting_Noun_Chunks:
    
    
    def __init__(self):
        pass
    
    
    def preprocess_article(self,article):

            #-------Filtering using regex-------#
            # Steps followed:
            # 1. Passing in text in smallcase.
            # 2. Removing numbers.
            # 3. Removing any extra tabs.
            article = re.sub('\s\s+', ' ',(re.sub('[^ a-z]', ' ', article.lower())))


            #-------Extracting the words that actually have meaning - basically removing stopwords-------#
            # Steps:
            # 1. Pulling out distinct stopwords.
            # 2. Creating a list of words that are not stopwords.
            # 3. Converting the list to a string
            article = article.split(' ')
            article = " ".join([word for word in article if word not in set(stopwords.words('english'))])

            return article
    
    
    def extract_noun_chunks(self, article):
        
        #-------Creating a Spacy Document-------#
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(article)

        
        #-------Extracting Noun-Chunks based on combination of 2 or 3 words (Bigrams/Trigrams)-------#
        noun_chunks = [i.text for i in list(doc.noun_chunks) if (1 < len(i.text.split(' ')) < 4)]
        return noun_chunks
    
    
    def word_makes_sense(self, noun_chunks):
        
        n_chunk = []
        
        for word in noun_chunks:
            #---Splitting seperate words and creating a list---#
            word = word.split(' ')
            
            #---Checking if the word makes sense and adding the words that make sense to new list---#
            temp = [i for i in word if i in set(words.words())]
        
        #---Comparing new list to old list---#
            if len(temp) < len(word):
                #---If the length is same, i.e. all the words in the old list makes sense, then returning the word---# 
                n_chunk.append(' '.join(word))
        
        return n_chunk

    
    def postprocess_noun_chunks(self, noun_chunks):
        
        #-------Deduplicating Noun-Chunks and removing words that doesn't make sense-------#
        noun_chunks = self.word_makes_sense(list(set(noun_chunks)))
        
        return noun_chunks
    
    
    def tfidf_vectorizer(self, all_articles):
       
        #-------Calulating Term Frequency, Inverse Document Frequency (Tfidf) to get top noun-chunks-------#
        # Steps followed:
        # 1. Instantiating TfidfVectorizer().
        # 2. Fitting and transforming all the articles on the vectorizer.
        # 3. Calculating the scores.
        # 4. Getting feature words (sorted by index) selected from the raw documents
        vectorizer = TfidfVectorizer(ngram_range=(2,3))
        
        scores = vectorizer.fit_transform(all_articles)  # has a shape of (20, 19152)
        
        scores = scores.toarray().sum(axis=0)            # scores = summation of all the rows(20).

        total_words = vectorizer.get_feature_names()        
        
        return scores, total_words
    
    
    def __call__(self, list_of_articles):
        
        print('Please wait while the model finds the top 10 Noun-Chunks...\n')
        noun_chunks = []
        
        # Preprocessing and extracting Noun_Chunks
        for article in list_of_articles:
            
            #-----Pre-Procesing-----#
            article = self.preprocess_article(article)
            
            #-----Extracting Noun-Chunks-----#
            noun_chunks.extend(self.extract_noun_chunks(article))
            
            #-----Post-Procesing-----#            
            noun_chunks = self.postprocess_noun_chunks(noun_chunks)
            
            #-----Calculating tfidf score and extracting the total words-----#                        
            scores, total_words = self.tfidf_vectorizer(list_of_articles)

            #-----Creating a list of noun_chunks and their scores-----#
            noun_chunks_score = []
            for i in noun_chunks:
                try:
                    total_words_index = total_words.index(i)
                    noun_chunks_score.append([i, scores[total_words_index]])
                except:
                    pass
            
            #-----Sorting the score in descending order-----#
            nc_score = sorted(noun_chunks_score, key=lambda x:-x[1])  # -x[1] sorts the list in descending order
            
            #-----Returning the top 10 noun_chunks-----#
            return [i[0] for i in nc_score[:10]]

@app.route('/extract_nc', methods=['POST'])

def extract_nc():
    print(request.method)
    if request.method == 'POST':
        model = Extracting_Noun_Chunks()
        noun_chunks = model(list_of_articles)
        return render_template('base.html', data_sent=True, data=json.dumps({"Noun Chunks":{"nc": noun_chunks}}))
    return render_template('base.html', data_sent=False)

if __name__ == "__main__":
    app.run(debug=True)