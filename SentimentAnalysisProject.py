import numpy as np 
import pandas as pd 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
import re

data = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\pythonDS\\Module 8\\Sentiment.csv')
data = data[data["sentiment"] != "Neutral"]
data.reset_index(inplace=True)
data.drop("index",axis=1,inplace=True)

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
ps=PorterStemmer()

for i in range(len(data)):
    words=re.sub("[^A-Za-z]"," ",data["text"][i])
    words=words.lower()
    words=word_tokenize(words)
    words=[ps.stem(word) for word in words if word not in set(stopwords.words("english"))]
    sent=" ".join(words)
    data["text"][i]=sent
    
for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = 2000
tokenizer = Tokenizer(nb_words=max_fatures, split=' ')
tokenizer.fit_on_texts(data['tweet'].values)
X = tokenizer.texts_to_sequences(data['tweet'].values)
X = pad_sequences(X)

embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1], dropout=0.2))
model.add(LSTM(lstm_out, dropout_U=0.2, dropout_W=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

Y = pd.get_dummies(data['sentiment']).values
xtrain, xtest, ytrain, ytest = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(xtrain.shape,ytrain.shape)
print(xtest.shape,ytest.shape)

batch_size = 32
model.fit(xtrain, ytrain, epochs = 10, batch_size=32, verbose = 2)


score,acc=model.evaluate(xtest, ytest, verbose = 2, batch_size = 32)


print(score,acc)

Ypred=model.predict(xtest)
ypred=np.argmax(Ypred,1)
Ytest=ytest[0:,1]

from sklearn.metrics import classification_report,confusion_matrix
 
print(classification_report(Ytest,ypred))

print(confusion_matrix(Ytest,ypred))