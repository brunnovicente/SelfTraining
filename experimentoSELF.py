import warnings
warnings.filterwarnings("ignore")

import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import time
from sklearn.utils import shuffle
from SelfTraining import StandardSelfTraining

sca = MinMaxScaler()

dados = pd.read_csv('d:/basedados/agricultura.csv')
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values

X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.9, test_size=0.1, stratify=Y)


dados = pd.DataFrame(X)
dados['classe'] = Y
rotulados = [50 , 100, 150, 200, 250, 300]
porcentagem = [0.0047, 0.0093, 0.0140, 0.0186, 0.0233, 0.0279]

rotulados = [50, 100, 150, 200, 250, 300]
porcentagem = [0.0047, 0.0093, 0.0140, 0.0186, 0.0233, 0.0279]

for r, p in enumerate(porcentagem):
    
    resultadoMLP = pd.DataFrame()
    resultadoKNN = pd.DataFrame()
    resultadoSVM = pd.DataFrame()
    resultadoRF = pd.DataFrame()
    resultadoNB = pd.DataFrame()
    resultadoLR = pd.DataFrame()
    
    resultadoMLP_T = pd.DataFrame()
    resultadoKNN_T = pd.DataFrame()
    resultadoSVM_T = pd.DataFrame()
    resultadoRF_T = pd.DataFrame()
    resultadoNB_T = pd.DataFrame()
    resultadoLR_T = pd.DataFrame()
    inicio = time.time()
    
    for k in np.arange(10):
        print('Teste: '+str(rotulados[r])+' - '+str(k+1))
        
        X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.9, test_size=0.1, stratify=Y)
                
        """ PROCESSO TRANSDUTIVO """
        L, U, y, yu = train_test_split(X_train, y_train, train_size = p, test_size= 1.0 - p, stratify=y_train)
        
        
       
        mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=100)
        knn = KNeighborsClassifier(n_neighbors=5)
        svm = SVC(probability=True)
        rf = RandomForestClassifier(n_estimators=20)
        nb = GaussianNB()
        lr = LogisticRegression()
        
        self1 = StandardSelfTraining('MLP', mlp)
        self2 = StandardSelfTraining('KNN', knn)
        self3 = StandardSelfTraining('SVM', svm)
        self4 = StandardSelfTraining('RF', rf)
        self5 = StandardSelfTraining('NB', nb)
        self6 = StandardSelfTraining('LR', lr)
        
        X_treino = np.concatenate((L, U))
        Y_treino = np.concatenate((    
            y.astype(str),
            np.full_like(yu.astype(str), "unlabeled")
            ))
                
        self1.fit(X_treino, Y_treino)
        self2.fit(X_treino, Y_treino)
        self3.fit(X_treino, Y_treino)
        self4.fit(X_treino, Y_treino)
        self5.fit(X_treino, Y_treino)
        self6.fit(X_treino, Y_treino)
        
        """ PROCESSO TRANDUTIVO """
        
        resultadoMLP['exe'+str(k+1)] = self1.predict(U).astype('int64')
        resultadoMLP['y'+str(k+1)] = yu
        resultadoKNN['exe'+str(k+1)] = self2.predict(U).astype('int64')
        resultadoKNN['y'+str(k+1)] = yu
        resultadoSVM['exe'+str(k+1)] = self3.predict(U).astype('int64')
        resultadoSVM['y'+str(k+1)] = yu
        resultadoRF['exe'+str(k+1)] = self4.predict(U).astype('int64')
        resultadoRF['y'+str(k+1)] = yu
        resultadoNB['exe'+str(k+1)] = self5.predict(U).astype('int64')
        resultadoNB['y'+str(k+1)] = yu
        resultadoLR['exe'+str(k+1)] = self6.predict(U).astype('int64')
        resultadoLR['y'+str(k+1)] = yu
        
        
        """ PROCESSO INDUTIVO """
        resultadoMLP_T['exe'+str(k+1)] = self1.predict(X_test).astype('int64')
        resultadoMLP_T['y'+str(k+1)] = y_test
        resultadoKNN_T['exe'+str(k+1)] = self2.predict(X_test).astype('int64')
        resultadoKNN_T['y'+str(k+1)] = y_test
        resultadoSVM_T['exe'+str(k+1)] = self3.predict(X_test).astype('int64')
        resultadoSVM_T['y'+str(k+1)] = y_test
        resultadoRF_T['exe'+str(k+1)] = self4.predict(X_test).astype('int64')
        resultadoRF_T['y'+str(k+1)] = y_test
        resultadoNB_T['exe'+str(k+1)] = self5.predict(X_test).astype('int64')
        resultadoNB_T['y'+str(k+1)] = y_test
        resultadoLR_T['exe'+str(k+1)] = self6.predict(X_test).astype('int64')
        resultadoLR_T['y'+str(k+1)] = y_test
        
    fim = time.time()
    tempo = np.round((fim - inicio)/60,2)
    print('........ Tempo: '+str(tempo)+' minutos.')
                    
    resultadoMLP.to_csv('resultados/resultado_SELF_MLP_'+str(rotulados[r])+'.csv', index=False)
    resultadoKNN.to_csv('resultados/resultado_SELF_KNN_'+str(rotulados[r])+'.csv', index=False)
    resultadoSVM.to_csv('resultados/resultado_SELF_SVM_'+str(rotulados[r])+'.csv', index=False)
    resultadoRF.to_csv('resultados/resultado_SELF_RF_'+str(rotulados[r])+'.csv', index=False)
    resultadoNB.to_csv('resultados/resultado_SELF_NB_'+str(rotulados[r])+'.csv', index=False)
    resultadoLR.to_csv('resultados/resultado_SEKF_LR_'+str(rotulados[r])+'.csv', index=False)
    
    resultadoMLP_T.to_csv('resultados/resultado_SELF_MLP_T'+str(rotulados[r])+'.csv', index=False)
    resultadoKNN_T.to_csv('resultados/resultado_SELF_KNN_T'+str(rotulados[r])+'.csv', index=False)
    resultadoSVM_T.to_csv('resultados/resultado_SELF_SVM_T'+str(rotulados[r])+'.csv', index=False)
    resultadoRF_T.to_csv('resultados/resultado_SELF_RF_T'+str(rotulados[r])+'.csv', index=False)
    resultadoNB_T.to_csv('resultados/resultado_SELF_NB_T'+str(rotulados[r])+'.csv', index=False)
    resultadoLR_T.to_csv('resultados/resultado_SEKF_LR_T'+str(rotulados[r])+'.csv', index=False)