import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report

# Carregar o dataset CSV do Kaggle
data = pd.read_csv('diabetes.csv')

# Separar os dados em recursos (X) e rótulos (y)
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Dividir o dataset em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar os dados (Padronização)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Inicializar o modelo Perceptron
perceptron = Perceptron()

# Treinar o modelo
perceptron.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = perceptron.predict(X_test)

# Avaliar o desempenho do modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("Relatório de Classificação:\n", classification_report(y_test, y_pred))
