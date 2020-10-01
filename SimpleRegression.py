# Imprtacao de bibliotecas do python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Definicao do dataset a ser utilizado

dataset = pd.read_csv('C:/Users/allan/OneDrive/Área de Trabalho/Udemy - Curso/Salary_Data.csv')
print(dataset)

# Divisao entre variavel dependente e independente

x = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values

# Importacao da bliblioteca de traino e teste e separacao de variaves para treino e utras para teste

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Biblioteca utilizada para a criacao de uma Regressao linear

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train) # Integracao entre as variaveis dependente e independente com a regressao simples

# Variavel de predicao da regressao

y_pred = regressor.predict(x_test)

# Criacao do grafico de treino

plt.scatter(x_train, y_train, color = 'red') # Parte dos pontos vermelhos do grafico em que mostra o dado correto
plt.plot(x_train, regressor.predict(x_train), color = 'blue') # Parte da linha azul do grafico em que demonstra a predicao
plt.title('Salary Vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Criacao do grafico de predicao porem da parte de teste

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary Vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# Como prever um dado que esta fora do dataset, como por exemplo uma pessao que tem 12 anos de experiencia

print(regressor.predict([[12]]))