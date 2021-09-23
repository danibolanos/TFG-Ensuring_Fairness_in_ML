"""
Replicación en Python del caso de estudio del artículo Counterfactual Fairness (Kusner et al. 2017)
"""

import pystan
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from pathlib import Path

# function to convert to a dictionary for use with STAN train-time model
def get_pystan_train_dic(pandas_df, sense_cols):
    dic_out = {}
    dic_out['N'] = len(pandas_df)
    dic_out['C'] = len(sense_cols)
    dic_out['A'] = np.array(pandas_df[sense_cols])
    dic_out['UGPA'] = list(pandas_df['UGPA'])
    dic_out['LSAT'] = list(pandas_df['LSAT'])
    dic_out['ZFYA'] = list(pandas_df['ZFYA'])
    return dic_out

# function to convert to a dictionary for use with STAN test-time model
def get_pystan_test_dic(fit_extract, test_dic):
    dic_out = {}
    for key in fit_extract.keys():
        if key not in ['sigma2_G', 'K', 'wA_F', 'wK_F']:
            dic_out[key] = np.mean(fit_extract[key], axis=0)
    
    need_list = ['N', 'C', 'A', 'UGPA', 'LSAT']
    for data in need_list:
        dic_out[data] = test_dic[data]

    return dic_out

# Preprocesamiento del conjunto de datos 'law_data.csv'
def preprocess_data():
    law_data = pd.read_csv('./datos/law_data.csv', index_col=0)  

    law_data = pd.get_dummies(law_data, columns=['race'], prefix='', prefix_sep='')

    sense_cols = ['Amerindian','Asian','Black','Hispanic','Mexican','Other','Puertorican','White','Male','Female']

    law_data['Female'] = law_data['sex'].apply(lambda a: 1 if a == 1 else 0)
    law_data['Male'] = law_data['sex'].apply(lambda a: 1 if a == 2 else 0)
    law_data = law_data.drop(['sex'], axis=1)
    
    law_data['LSAT'] = law_data['LSAT'].apply(lambda a: int(a))

    law_train, law_test = train_test_split(law_data, random_state = 76592621, train_size = 0.8);

    law_train_dic = get_pystan_train_dic(law_train, sense_cols)
    law_test_dic = get_pystan_train_dic(law_test, sense_cols)

    return law_train_dic, law_test_dic
    
# Modelo Total: usa todos los atributos para la predicción
def mod_full(law_train_dic, law_test_dic):
    # Construcción de los conjuntos de entrenamiento y tests para el modelo
    x_train = np.hstack((law_train_dic['A'], np.array(law_train_dic['UGPA']).reshape(-1,1), 
                         np.array(law_train_dic['LSAT']).reshape(-1,1)))
    x_test = np.hstack((law_test_dic['A'], np.array(law_test_dic['UGPA']).reshape(-1,1), 
                        np.array(law_test_dic['LSAT']).reshape(-1,1)))
    y = law_train_dic['ZFYA']

    # Entrenamiento del modelo sobre el conjunto x_train
    lr_full = LinearRegression()
    lr_full.fit(x_train, y)
    
    # Predicción de las etiquetas sobre el conjunto x_test
    preds = lr_full.predict(x_test)

    return preds

# Modelo equidad por desconocimiento: no usa los atributos sensibles en predicción
def mod_unaware(law_train_dic, law_test_dic):
    # Construcción de los conjuntos de entrenamiento y tests para el modelo
    x_train = np.hstack((np.array(law_train_dic['UGPA']).reshape(-1,1), 
                         np.array(law_train_dic['LSAT']).reshape(-1,1)))
    x_test = np.hstack((np.array(law_test_dic['UGPA']).reshape(-1,1), 
                        np.array(law_test_dic['LSAT']).reshape(-1,1)))
    y = law_train_dic['ZFYA']
    
    # Entrenamiento del modelo sobre el conjunto x_train
    lr_unaware = LinearRegression()
    lr_unaware.fit(x_train, y)
    
    # Predicción de las etiquetas sobre el conjunto x_test
    preds = lr_unaware.predict(x_test)

    return preds

# Modelo no determinista: suponemos variable de ruido 'K' para generar la distribución del resto
def mod_fair_k(law_train_dic, law_test_dic):

    #check_fit = Path("./model_fit.pkl")

    # Compile Model
    model = pystan.StanModel(file = './datos/stan/law_school_train.stan')
    print('Finished compiling model!')
    # Commence the training of the model to infer weights (500 warmup, 500 actual)
    fit = model.sampling(data = law_train_dic, seed=76592621, chains=1, iter=2)
    post_samps = fit.extract()

    # Retreive posterior weight samples and take means
    law_train_dic_final = get_pystan_test_dic(post_samps, law_train_dic)
    law_test_dic_final = get_pystan_test_dic(post_samps, law_test_dic)

    #check_train = Path("./model_fit_train.pkl")
    
    # Obtain posterior training samples from scratch
    model_train = pystan.StanModel(file = './datos/stan/law_school_only_u.stan')
    fit_train = model_train.sampling(data = law_train_dic_final, seed=76592621, chains=1, iter=2)
    fit_train_samps = fit_train.extract()
    
    x_train = np.mean(fit_train_samps['K'], axis=0).reshape(-1,1)

    #check_test = Path("./model_fit_test.pkl")


    # Obtain posterior test samples from scratch
    model_test = pystan.StanModel(file = './datos/stan/law_school_only_u.stan')
    fit_test = model_test.sampling(data = law_test_dic_final, seed=76592621, chains=1, iter=2)
    fit_test_samps = fit_test.extract()
    
    x_test = np.mean(fit_test_samps['K'], axis=0).reshape(-1,1)
    y = law_train_dic['ZFYA']

    lr_fair_k = LinearRegression()
    lr_fair_k.fit(x_train, y)
    
    preds = lr_fair_k.predict(x_test)

    # Return Results:
    return preds
    
if __name__ == '__main__':  

    # Obtiene en un diccionario el conjunto de datos procesado y en una partición 80 (train) 20 (test)
    law_train_dic, law_test_dic = preprocess_data()

    # Obtiene las predicciones para cada modelo
    preds_full = mod_full(law_train_dic, law_test_dic)
    preds_unaware = mod_unaware(law_train_dic, law_test_dic)
    preds_fair_k = mod_fair_k(law_train_dic, law_test_dic)

    # Imprime las predicciones resultantes
    print('Full: %.3f' % np.sqrt(mean_squared_error(preds_full, law_test_dic['ZFYA'])))
    print('Unaware: %.3f' % np.sqrt(mean_squared_error(preds_unaware, law_test_dic['ZFYA'])))
    print('Fair K: %.3f' % np.sqrt(mean_squared_error(preds_fair_k,law_test_dic['ZFYA'])))
