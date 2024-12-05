# importing all libs
import pickle
import re
from collections import Counter
from itertools import chain

import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.metrics import mean_absolute_error as mse
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor

import matplotlib.pyplot as plt

from util import shortWord_Dictionary, load_glove

# loading two different datasets
# considering apple stock data as primary data
dataset = pd.read_csv("./AppleNewsStock.csv")

# creating a list of stopwords
stop = stopwords.words('english')

# shape of the dataset
print(dataset.shape)

# separating dataset into two sets as stock data and news data
stockData = dataset.loc[:, "Date":"Volume"]
newsData = dataset[['Date', "News"]]

# checking for any value
print()
print("Stock data result : ", stockData.isnull().any(), sep="\n")
print("News data result : ", newsData.isnull().any(), sep="\n")

# filling the null values in news data
newsData.fillna(method='ffill', inplace=True)

# considering one day value by taking the difference betweem next day
stockData = stockData.set_index('Date').diff(periods=1)
stockData['Date'] = stockData.index
stockData = stockData.reset_index(drop=True)

# dropping unwanted columns since we need only open data
stockData = stockData.drop(['High', 'Low', 'Close', 'Volume', 'Adj Close'], 1)

# dropping null rows
stockData = stockData[stockData.Open.notnull()]
print()
print(stockData.head())

# creating new data where date and corresponding news along with opening price
news = []
openingPrice = []

for row in stockData.iterrows():
    openingPrice.append(row[1]['Open'])
    headline = newsData[newsData['Date'] == row[1]["Date"]]
    news.append(headline['News'].item())

# length of the price and news
print()
print("News :", len(news), "Opening Price :", len(openingPrice))


# function for cleaning the text
def cleaning(raw):
    # replacing shortwords in text
    shortword_free = [shortWord_Dictionary[word] if word in shortWord_Dictionary else word for word in
                      raw.lower().split()]
    # removing special characters
    only_alphanumeric = re.sub("[^a-zA-Z0-9]", " ", ' '.join(shortword_free))
    # removing stop words from the text
    stopword_free = [word for word in only_alphanumeric.split() if word not in stop]
    return " ".join(stopword_free)


# printing the uncleaned news and cleaned news.
print()
print("Uncleaned :- ", news[0])
print("Cleaned  :- ", cleaning(news[0]))

cleanedNews = list(map(cleaning, news))

# counting occurrence of each word
mergedNews = list(chain.from_iterable(list(map(str.split, cleanedNews))))
wordFreq = dict(Counter(mergedNews).most_common())
print()
# print(wordFreq)
print("Total Number of words :", len(wordFreq))

# loading 300d glove file for word embedding
word2emb = load_glove()
print()
print("Total Number of embedding words : ", len(word2emb))

# we are going to use the words which occurred more than 10 and create the word to vector dictionary
word2int = dict()
threshold = 10
cnt = 0

for words, count in wordFreq.items():
    if count >= threshold or words in word2emb:
        word2int[words] = cnt
        cnt += 1

# for the words which are less than threshold and not in embedding list we will assign unk and pad
codes = ["UNK", "PAD"]
for code in codes:
    word2int[code] = len(word2int)

# reversing the word to vec to have vec to word dictionary
int2word = dict((vec, word) for word, vec in word2int.items())

print()
print("Total Number of Words:", len(wordFreq))
print("Number of Words converted into vector:", len(word2int))
print("Percent of Words we will use: {}%".format(round(len(word2int) / len(wordFreq), 4) * 100))

# creating word embedding matrix for the news data
embeddingDimension = 300
wordsCount = len(word2int)
embeddingMatrix = np.zeros((wordsCount, embeddingDimension))

for word, vec in word2int.items():
    if word in word2emb:
        embeddingMatrix[vec] = word2emb[word]
    else:
        # if a given word not in the glove embedding, we will initialize the vector
        tempEmbedding = np.array(np.random.uniform(-1.0, 1.0, embeddingDimension))
        word2emb[word] = tempEmbedding
        embeddingMatrix[vec] = tempEmbedding

# vectorization of news data
word_count = 0
unk_count = 0

newsData_vectorized = list()

for data in cleanedNews:
    temp = []
    for word in data.split():
        if word in word2int:
            temp.append(word2int[word])
        else:
            temp.append(word2int['UNK'])
    newsData_vectorized.append(temp)
np.save("word_2_vector.npy",word2int)

# in order to avoid long training time, we are going to trim the dataset to 200 words
wordLimit = 200
newHeadlinesData = []
for dailyNews in newsData_vectorized:
    if len(dailyNews) >= wordLimit:
        newHeadlinesData.append(dailyNews[:wordLimit])
    else:
        for i in range(wordLimit - len(dailyNews)):
            dailyNews.append(word2int['PAD'])
        newHeadlinesData.append(dailyNews)

# time to normalize opening price
minCost = min(openingPrice)
maxCost = max(openingPrice)
misc = dict()
misc['minCost'] = minCost
misc['maxCost'] = maxCost
np.save("./misc.npy",misc)

def normalize(cost):
    return (cost - minCost) / (maxCost - minCost)

normalizedOpeningprice = list(map(normalize, openingPrice))

print()
print('Before normalization :- ')
print(minCost)
print(maxCost)
print(np.mean(openingPrice))

print()
print('After normalization :- ')
print(min(normalizedOpeningprice))
print(max(normalizedOpeningprice))
print(np.mean(normalizedOpeningprice))

# splitting the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(newHeadlinesData, normalizedOpeningprice, test_size=0.2,
                                                    random_state=3)

# converting into array
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print()
print("Train data shape : ", x_train.shape)
print("Test data shape : ", x_test.shape)

# defining models
LR = LinearRegression()
SVM = SVR()
RF = RandomForestRegressor()
KNN = KNeighborsRegressor()
AB = AdaBoostRegressor()
GB = GradientBoostingRegressor()


# function for training and prediction
def train_predict(model, trainX, trainY, testX, testY):
    model.fit(trainX, trainY)
    y_pred = model.predict(testX)
    acc_test = mse(testY, y_pred)
    return model, acc_test, y_pred


_, LR_acc, _ = train_predict(LR, x_train, y_train, x_test, y_test)
print("The error score of Linear Regression is %f" % LR_acc)
_, SVM_acc, _ = train_predict(SVM, x_train, y_train, x_test, y_test)
print("The error score of SVM is %f" % SVM_acc)
_, RF_acc, _ = train_predict(RF, x_train, y_train, x_test, y_test)
print("The error score of Random Forest Regressior   is %f" % RF_acc)
_, KNN_acc, _ = train_predict(KNN, x_train, y_train, x_test, y_test)
print("The error score of K-nearest neighbour is %f" % KNN_acc)
_, AB_ac, _ = train_predict(AB, x_train, y_train, x_test, y_test)
print("The error score of adaboosting is %f" % AB_ac)
GB_model, GB_ac, predictions = train_predict(GB, x_train, y_train, x_test, y_test)
print("The error score of gradient boosting is %f" % GB_ac)


def reverse_normalization(cost):
    return cost * (maxCost - minCost) + minCost


reverNormalized_prediction = list(map(reverse_normalization, predictions))
reverNormalized_Ytest = list(map(reverse_normalization, y_test))

# Plot the predicted (blue) and actual (green) values
plt.figure(figsize=(12, 4))
plt.plot(reverNormalized_prediction, 'blue')
plt.plot(reverNormalized_Ytest, 'green')
plt.title("Predicted (blue) vs Actual (green) Opening Price Changes")
plt.xlabel("Instances")
plt.ylabel("Change in Opening Price")
plt.show()

pickle.dump(GB_model, open("GB_model.pkl", "wb"))
