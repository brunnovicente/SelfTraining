import pandas as pd
import numpy as np

modelos = ['knn', 'mlp', 'svm', 'rf']
bases = ['mnist', 'fashion', 'usps', 'cifar10', 'slt10', 'reuters', 'epilepsia', 'covtype']

for modelo in modelos:
    print(modelo)
    for base in bases:
        print('.... '+base)
        dados = pd.read_csv(modelo+'/'+base+'.csv')
        resultado = pd.DataFrame()
        resultado['R'] = dados['R']
        resultado['AT'] = dados['AT']
        resultado['AI'] = (dados['AT'] + dados['KT'] + dados['KI'])/3
        resultado['KT'] = dados['KT']
        resultado['KI'] = dados['KI']
        
        if(base == 'slt10'):
            base= 'stl10'
            
        resultado.to_csv('D:/Drive UFRN/Doutorado/Resultados/Artigo KBS/compilados/resultado_TRI_'+modelo+'_'+base+'.csv', index=False)
