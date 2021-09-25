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

# Crea un diccionario compatible con pystan para el conjunto de datos y los atributos protegidos dados
def create_pystan_dic(data, protected_attr):
    dic_data = {}
    dic_data['N'] = len(data)
    dic_data['C'] = len(protected_attr)
    dic_data['A'] = np.array(data[protected_attr])
    dic_data['UGPA'] = list(data['UGPA'])
    dic_data['LSAT'] = list(data['LSAT'])
    dic_data['ZFYA'] = list(data['ZFYA'])
    return dic_data


# Preprocesamiento del conjunto de datos 'law_data.csv'
def preprocess_data():
    # Leemos el conjunto de datos con el nombre pasado por parámetro
    dataset = pd.read_csv('./datos/law_data.csv', index_col=0)  
    
    # Separamos cada atributo de raza en una columna con 1 si el individuo 
    # pertenece a ella o 0 si el individuo no pertenece
    dataset = pd.get_dummies(dataset, columns=['race'], prefix='', prefix_sep='')

    # Creamos una columna que indique con 0 o 1 la pertenencia al sexo Masculino o Femenino
    dataset['Female'] = dataset['sex'].apply(lambda a: 1 if a == 1 else 0)
    dataset['Male'] = dataset['sex'].apply(lambda a: 1 if a == 2 else 0)
    dataset = dataset.drop(['sex'], axis=1)
    
    # Guardamos en un vector todos los atributos protegidos
    protected_attr = ['Amerindian','Asian','Black','Hispanic','Mexican','Other','Puertorican'
                      ,'White','Male','Female']
    
    # Convertimos la columna 'LSAT' a tipo entero
    dataset['LSAT'] = dataset['LSAT'].apply(lambda a: int(a))

    # Realizamos una división 80-20 de los conjuntos de entrenamiento y test
    train, test = train_test_split(dataset, random_state = 76592621, train_size = 0.8);

    # Creamos un diccionario compatible con pystan para los conjuntos creados anteriormente
    dic_train = create_pystan_dic(train, protected_attr)
    dic_test = create_pystan_dic(test, protected_attr)
    dic_full = create_pystan_dic(dataset, protected_attr)

    return dic_train, dic_test, dic_full

    
# Modelo Total: usa todos los atributos para la predicción
def mod_full(dic_train, dic_test):
    # Construcción de los conjuntos de entrenamiento y tests para el modelo
    x_train = np.hstack((dic_train['A'], np.array(dic_train['UGPA']).reshape(-1,1), 
                         np.array(dic_train['LSAT']).reshape(-1,1)))
    x_test = np.hstack((dic_test['A'], np.array(dic_test['UGPA']).reshape(-1,1), 
                        np.array(dic_test['LSAT']).reshape(-1,1)))
    y = dic_train['ZFYA']

    # Entrenamiento del modelo sobre el conjunto x_train
    lr_full = LinearRegression()
    lr_full.fit(x_train, y)
    
    # Predicción de las etiquetas sobre el conjunto x_test
    preds = lr_full.predict(x_test)

    return preds

# Modelo equidad por desconocimiento: no usa los atributos sensibles en predicción
def mod_unaware(dic_train, dic_test):
    # Construcción de los conjuntos de entrenamiento y tests para el modelo
    x_train = np.hstack((np.array(dic_train['UGPA']).reshape(-1,1), 
                         np.array(dic_train['LSAT']).reshape(-1,1)))
    x_test = np.hstack((np.array(dic_test['UGPA']).reshape(-1,1), 
                        np.array(dic_test['LSAT']).reshape(-1,1)))
    y = dic_train['ZFYA']
    
    # Entrenamiento del modelo sobre el conjunto x_train
    lr_unaware = LinearRegression()
    lr_unaware.fit(x_train, y)
    
    # Predicción de las etiquetas sobre el conjunto x_test
    preds = lr_unaware.predict(x_test)

    return preds

# Creamos un diccionario con la media de los parámetros útiles para el modelo de 'K'
# obtenidos en el entrenamiento previo para el modelo base
def get_useful_param(samples, dic):
    dic_data = {}
    # Añadimos los parámetros comunes que comparte con el diccionario original    
    param_base = ['N', 'C', 'A', 'UGPA', 'LSAT']
    for param in param_base:
        dic_data[param] = dic[param]
        
    # Guardamos la media del vector de valores para los parámetros que utiliza el modelo '*only_k.stan'
    for key in samples.keys():
        if key not in ['K', 'wK_F', 'wA_F', 'sigma2_G', 'lp__']:
            dic_data[key] = np.mean(samples[key], axis=0)

    return dic_data

# Entrenamos el modelo para 'K' para el diccionario dado
def k_model(dic_post):
    
    #check_train = Path("./model_fit_train/test.pkl")
    
    # Obtiene muestras desde cero para la variable 'K' a partir del modelo 'law_school_only_k.stan'
    model = pystan.StanModel(file = './datos/stan/law_school_only_k.stan')
    fit_data = model.sampling(data = dic_post, seed=76592621, chains=1, iter=2)
    fit_samples = fit_data.extract()
    # Realiza la media de las muestras de 'K'
    x = np.mean(fit_samples['K'], axis=0).reshape(-1,1)
    
    return x

# Modelo no determinista: suponemos variable de ruido 'K' para generar la distribución del resto
def mod_fair_k(dic_train, dic_test):

    #check_fit = Path("./model_fit.pkl")

    # Compilamos el modelo de entrenamiento observado dado por el archivo 'law_school_train.stan'
    base_model = pystan.StanModel(file = './datos/stan/law_school_train.stan')
    # Entrenamos el modelo a partir de la función de muestreo (500 iter)
    fit_base = base_model.sampling(data = dic_train, seed=76592621, chains=1, iter=2)
    # Extraemos las muestras obtenidas por la inferencia sobre el modelo entrenado
    base_samples = fit_base.extract()

    # Utilizamos los parámetros útiles del entrenamiento previo para estimar la 
    # distribución posterior de 'K'
    dic_train_post = get_useful_param(base_samples, dic_train)
    dic_test_post = get_useful_param(base_samples, dic_test)

    # Obtenemos los conjuntos dde entrenamiento y prueba con la nueva variable 'K'
    x_train = k_model(dic_train_post)
    x_test = k_model(dic_test_post)
    y = dic_train['ZFYA']
    
    # Entrenamiento del modelo sobre el conjunto x_train
    lr_fair_k = LinearRegression()
    lr_fair_k.fit(x_train, y)
    
    # Predicción de las etiquetas sobre el conjunto x_test
    preds = lr_fair_k.predict(x_test)

    return preds

# Modelo determinista: añadimos términos de error aditivos independientes de los atributos protegidos
def mod_fair_add(dic_train, dic_test, dataset):
    # Construimos el conjunto total
    lr_eps_G = LinearRegression()
    lr_eps_G.fit(dataset['A'], dataset['UGPA'])
    eps_g_train = dic_train['UGPA'] - lr_eps_G.predict(dic_train['A'])
    eps_g_test = dic_test['UGPA'] - lr_eps_G.predict(dic_test['A'])
    
    # abduct the epsilon_L values
    lr_eps_l = LinearRegression()
    lr_eps_l.fit(dataset['A'], dataset['LSAT'])
    eps_l_train = dic_train['LSAT'] - lr_eps_l.predict(dic_train['A'])
    eps_l_test = dic_test['LSAT'] - lr_eps_l.predict(dic_test['A'])

    # predict on target using abducted latents
    lr_fair_add =  LinearRegression()
    lr_fair_add.fit(np.hstack((eps_g_train.reshape(-1,1),eps_l_train.reshape(-1,1))),dic_train['ZFYA'])

    # predict on test epsilons
    preds = lr_fair_add.predict(np.hstack((eps_g_test.reshape(-1,1),eps_l_test.reshape(-1,1))))

    return preds
    
if __name__ == '__main__':  

    # Obtiene en un diccionario el conjunto de datos y en una partición 80 (train) 20 (test)
    dic_train, dic_test, data_full = preprocess_data()

    # Obtiene las predicciones para cada modelo
    preds_full = mod_full(dic_train, dic_test)
    preds_unaware = mod_unaware(dic_train, dic_test)
    #preds_fair_k = mod_fair_k(dic_train, dic_test)
    preds_fair_add = mod_fair_add(dic_train, dic_test, data_full)

    # Imprime las predicciones resultantes
    print('Full: %.3f' % np.sqrt(mean_squared_error(preds_full, dic_test['ZFYA'])))
    print('Unaware: %.3f' % np.sqrt(mean_squared_error(preds_unaware, dic_test['ZFYA'])))
    #print('Fair K: %.3f' % np.sqrt(mean_squared_error(preds_fair_k, dic_test['ZFYA'])))
    print('Fair Add: %.3f' % np.sqrt(mean_squared_error(preds_fair_add, dic_test['ZFYA'])))
