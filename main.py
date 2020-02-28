import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from SelfTraining import StandardSelfTraining
from TriTraining import TriTraining
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


dados = pd.read_csv('d:/basedados/agricultura.csv')
dados = shuffle(dados)
X = dados.drop(['classe'], axis=1).values
Y = dados['classe'].values

L, U, y, yu = train_test_split(X,Y, train_size=0.1, test_size=0.9, stratify=Y)

X_train = np.concatenate((L, U))
y_train = np.concatenate((    
    y.astype(str),
    np.full_like(yu.astype(str), "unlabeled")
    ))

KNN = KNeighborsClassifier(n_neighbors=3, metric="euclidean")

selfTrainer = StandardSelfTraining('KNN', KNN)
selfTrainer.fit(X_train, y_train)

print('Tranduciteve score: ', selfTrainer.score(U, yu.astype(str)))
#print('Tranduciteve score: ', selfTrainer.score(L, y.astype(str)))
preditas = selfTrainer.predict(U).astype('int64')