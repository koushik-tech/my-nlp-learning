import nltk
# nltk.download('all')

# Tokenization using NLTK
from nltk import word_tokenize , sent_tokenize
sentence = "I am learning NLP, I need to utilise it in Data engineering."
print(f"word tokenize: {word_tokenize(sentence)}")
print(f"sent tokenize: {sent_tokenize(sentence)}")

# Stemming using NLTK
from nltk.stem import PorterStemmer
porter = PorterStemmer()
print(f"Stem PLAY : {porter.stem('PLAY')}")
print(f"Stem PLAYING : {porter.stem('PLAYING')}")
print(f"Stem PLAYER : {porter.stem('PLAYER')}")
print(f"Stem PLAYED : {porter.stem('PLAYED')}")
print(f"Stem PLAYS : {porter.stem('PLAYS')}")
print(f"Stem Communication : {porter.stem('Communication')}")
print(f"Stem International : {porter.stem('International')}")
print(f"Stem Functional : {porter.stem('Functional')}")
print(f"Stem interested : {porter.stem('interested')}")



# Lemmatization using NLTK

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
print(f"Lemmatizer PLAY : {lemmatizer.lemmatize("play","v")}")
print(f"Lemmatizer PLAYED : {lemmatizer.lemmatize("played","v")}")
print(f"Lemmatizer PLAYER : {lemmatizer.lemmatize("player","n")}")
print(f"Lemmatizer PLAYING : {lemmatizer.lemmatize("playing","v")}")
print(f"Lemmatizer Communication : {lemmatizer.lemmatize("Communication","v")}")
print(f"Lemmatizer International : {lemmatizer.lemmatize("International","v")}")
print(f"Lemmatizer interested : {lemmatizer.lemmatize("interested","v")}")


# POS Tagging using nltk

from nltk import pos_tag
from nltk import word_tokenize

text = "I am learning NLP slowly."
tokenized_text = word_tokenize(text)
print(f"tokenized text : {tokenized_text}")
tags = pos_tag(tokenized_text)
print(f"tags : {tags}")

# NER (Named Entity Recognition) using NLTK

from nltk import word_tokenize, pos_tag,ne_chunk

text = "Netaji Subhash chandra Bose was bron in Cuttak, Odisha. He is one of the greatest freedom fighter of India. He used to ride Honda City"
tokens = word_tokenize(text)
pos_tags = pos_tag(tokens)
named_entities = ne_chunk(pos_tags)
print(f"Named Entities : {named_entities}")

# Vectorization using NLTK

from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "I love NLP and love ML",
    "NLP is fun and interesting",
    "I hate boring stuff hate",
    "I hate him but you"
]

vectorizer = TfidfVectorizer()
x=vectorizer.fit_transform(corpus)
print(f"feature names: {vectorizer.get_feature_names_out()}")
print(x.toarray())

# Text Classification using NLTK

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

texts = ["I love this movie", "This film is terrible", "Absolutely great!", "Worst acting ever"]
labels = ["positive", "negative", "positive", "negative"]
vectorizer = TfidfVectorizer()
x=vectorizer.fit_transform(texts)
y=labels

x_train,x_test,y_train,y_test = train_test_split(x,y)
model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(classification_report(y_test,y_pred))
print(f"y_test: {y_test}")
print(f"y_pred: {y_pred}")
print(f"x_test: {x_test}")


# Prediction using NLTK

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = [
    "I love this movie",         # positive
    "This film is terrible",     # negative
    "Absolutely great!",         # positive
    "Worst acting ever",         # negative         
    "Excellent direction",       # positive
    "Not worth watching",        # negative
    "Amazing experience",        # positive
    "Bad script and poor acting", # negative
    "poor performance",  # negative
    "fabulous movie",  # positive
    "waste of time",  # negative
    "movie of the year"  # positive

]

labels = ["positive", "negative", "positive", "negative", "positive", "negative", "positive", "negative","negative", "positive", "negative", "positive"]

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(x,labels)

def predict_sentiment(sentence):
    sentence_lower = sentence.lower()
    vector = vectorizer.transform([sentence_lower])
    prediction = model.predict(vector)[0]
    return prediction

while True:
    user_input = input("Enter a review :")
    print(f"user_input: {user_input}")
    if user_input.lower().strip() =="exit" :
        break
    result = predict_sentiment(user_input)
    print(f"Predicted Sentiment : {result}")
