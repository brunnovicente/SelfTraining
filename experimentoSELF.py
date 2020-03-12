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
from sklearn.metrics import accuracy_score, cohen_kappa_score

sca = MinMaxScaler()
base = 'mnist'
caminho = 'D:/Drive UFRN/bases/'
dados = pd.read_csv(caminho + base +'.csv')
X = sca.fit_transform(dados.drop(['classe'], axis=1).values)
Y = dados['classe'].values

X_train, X_test, y_train, y_test = train_test_split(X,Y, train_size=0.9, test_size=0.1, stratify=Y)

dados = pd.DataFrame(X)
dados['classe'] = Y
rotulados = [50 , 100, 150, 200, 250, 300]
porcentagem = [0.0047, 0.0093, 0.0140, 0.0186, 0.0233, 0.0279]

rotulados = [50, 100, 150, 200, 250, 300]
porcentagem = [0.0047, 0.0093, 0.0140, 0.0186, 0.0233, 0.0279]

resultadoMLPi = pd.DataFrame()
resultadoKNNi = pd.DataFrame()
resultadoSVMi = pd.DataFrame()
resultadoRFi = pd.DataFrame()
resultadoNBi = pd.DataFrame()
resultadoLRi = pd.DataFrame()

resultadoMLPt = pd.DataFrame()
resultadoKNNt = pd.DataFrame()
resultadoSVMt = pd.DataFrame()
resultadoRFt = pd.DataFrame()
resultadoNBt = pd.DataFrame()
resultadoLRt = pd.DataFrame()

acuraciaMLPi = []
acuraciaKNNi = []
acuraciaSVMi = []
acuraciaRFi = []
acuraciaNBi = []
acuraciaLRi = []

kappaMLPi = []
kappaKNNi = []
kappaSVMi = []
kappaRFi = []
kappaNBi = []
kappaLRi = []

acuraciaMLPt = []
acuraciaKNNt = []
acuraciaSVMt = []
acuraciaRFt = []
acuraciaNBt = []
acuraciaLRt = []

kappaMLPt = []
kappaKNNt = []
kappaSVMt = []
kappaRFt = []
kappaNBt = []
kappaLRt = []

for r, p in enumerate(porcentagem):
    
    
    inicio = time.time()
    
    
    print('Teste: '+str(rotulados[r]))
    
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
    Y_treino = np.concatenate((y.astype(str), np.full_like(yu.astype(str), "unlabeled")))
            
    self1.fit(X_treino, Y_treino)
    self2.fit(X_treino, Y_treino)
    self3.fit(X_treino, Y_treino)
    self4.fit(X_treino, Y_treino)
    self5.fit(X_treino, Y_treino)
    self6.fit(X_treino, Y_treino)
    
    """ PROCESSO TRANDUTIVO """
    acuraciaMLPi.append(accuracy_score(yu, self1.predict(U).astype('int64')))
    acuraciaKNNi.append(accuracy_score(yu, self2.predict(U).astype('int64')))
    acuraciaSVMi.append(accuracy_score(yu, self3.predict(U).astype('int64')))
    acuraciaRFi.append(accuracy_score(yu, self4.predict(U).astype('int64')))
    acuraciaNBi.append(accuracy_score(yu, self5.predict(U).astype('int64')))
    acuraciaLRi.append(accuracy_score(yu, self6.predict(U).astype('int64')))
   
        
    kappaMLPi.append(cohen_kappa_score(yu, self1.predict(U).astype('int64')))
    kappaKNNi.append(cohen_kappa_score(yu, self2.predict(U).astype('int64')))
    kappaSVMi.append(cohen_kappa_score(yu, self3.predict(U).astype('int64')))
    kappaRFi.append(cohen_kappa_score(yu, self4.predict(U).astype('int64')))
    kappaNBi.append(cohen_kappa_score(yu, self5.predict(U).astype('int64')))
    kappaLRi.append(cohen_kappa_score(yu, self6.predict(U).astype('int64')))
    
    acuraciaMLPt.append(accuracy_score(y_test, self1.predict(X_test).astype('int64')))
    acuraciaKNNt.append(accuracy_score(y_test, self2.predict(X_test).astype('int64')))
    acuraciaSVMt.append(accuracy_score(y_test, self3.predict(X_test).astype('int64')))
    acuraciaRFt.append(accuracy_score(y_test, self4.predict(X_test).astype('int64')))
    acuraciaNBt.append(accuracy_score(y_test, self5.predict(X_test).astype('int64')))
    acuraciaLRt.append(accuracy_score(y_test, self6.predict(X_test).astype('int64')))
    
    kappaMLPt.append(cohen_kappa_score(y_test, self1.predict(X_test).astype('int64')))
    kappaKNNt.append(cohen_kappa_score(y_test, self2.predict(X_test).astype('int64')))
    kappaSVMt.append(cohen_kappa_score(y_test, self3.predict(X_test).astype('int64')))
    kappaRFt.append(cohen_kappa_score(y_test, self4.predict(X_test).astype('int64')))
    kappaNBt.append(cohen_kappa_score(y_test, self5.predict(X_test).astype('int64')))
    kappaLRt.append(cohen_kappa_score(y_test, self6.predict(X_test).astype('int64')))    
        
    fim = time.time()
    tempo = np.round((fim - inicio)/60,2)
    print('........ Tempo: '+str(tempo)+' minutos.')

resultadoMLPi['R'] = rotulados
resultadoKNNi['R'] = rotulados
resultadoSVMi['R'] = rotulados
resultadoRFi['R'] = rotulados
resultadoNBi['R'] = rotulados
resultadoLRi['R'] = rotulados

resultadoMLPt['R'] = rotulados
resultadoKNNt['R'] = rotulados
resultadoSVMt['R'] = rotulados
resultadoRFt['R'] = rotulados
resultadoNBt['R'] = rotulados
resultadoLRt['R'] = rotulados

resultadoMLPi['ACURACIA'] = acuraciaMLPi
resultadoKNNi['ACURACIA'] = acuraciaKNNi
resultadoSVMi['ACURACIA'] = acuraciaMLPi
resultadoRFi['ACURACIA'] = acuraciaRFi
resultadoNBi['ACURACIA'] = acuraciaNBi
resultadoLRi['ACURACIA'] = acuraciaLRi

resultadoMLPi['KAPPA'] = kappaMLPi
resultadoKNNi['KAPPA']  = kappaKNNi
resultadoSVMi['KAPPA']  = kappaSVMi
resultadoRFi['KAPPA']  = kappaRFi
resultadoNBi['KAPPA']  = kappaNBi
resultadoLRi['KAPPA']  = kappaLRi

resultadoMLPt['ACURACIA'] = acuraciaMLPt
resultadoKNNt['ACURACIA'] = acuraciaKNNt
resultadoSVMt['ACURACIA'] = acuraciaMLPt
resultadoRFt['ACURACIA'] = acuraciaRFt
resultadoNBt['ACURACIA'] = acuraciaNBt
resultadoLRt['ACURACIA'] = acuraciaLRt

resultadoMLPt['KAPPA'] = kappaMLPt
resultadoKNNt['KAPPA']  = kappaKNNt
resultadoSVMt['KAPPA']  = kappaSVMt
resultadoRFt['KAPPA']  = kappaRFt
resultadoNBt['KAPPA']  = kappaNBt
resultadoLRt['KAPPA']  = kappaLRt

resultadoMLPi.to_csv('resultados/resultado_SELF_MLP_i_'+base+'.csv', index=False)
resultadoKNNi.to_csv('resultados/resultado_SELF_KNN_i_'+base+'.csv', index=False)
resultadoSVMi.to_csv('resultados/resultado_SELF_SVM_i_'+base+'.csv', index=False)
resultadoRFi.to_csv('resultados/resultado_SELF_RF_i_'+base+'.csv', index=False)
resultadoNBi.to_csv('resultados/resultado_SELF_NB_i_'+base+'.csv', index=False)
resultadoLRi.to_csv('resultados/resultado_SEKF_LR_i_'+base+'.csv', index=False)

resultadoMLPt.to_csv('resultados/resultado_SELF_MLP_t'+base+'.csv', index=False)
resultadoKNNt.to_csv('resultados/resultado_SELF_KNN_t'+base+'.csv', index=False)
resultadoSVMt.to_csv('resultados/resultado_SELF_SVM_t'++base+'.csv', index=False)
resultadoRFt.to_csv('resultados/resultado_SELF_RF_t'+base+'.csv', index=False)
resultadoNBt.to_csv('resultados/resultado_SELF_NB_t'+base+'.csv', index=False)
resultadoLRt.to_csv('resultados/resultado_SEKF_LRt_'+base+'.csv', index=False)