import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def read_csv(path):
    df = pd.read_csv(path)

    #print(df.head())
    return df

def save_to_csv(df,path):
    df.to_csv(path, index = False)

def number_pattern(df,column):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df[column].fillna(df[column].median(), inplace = True)
    return df

def one_hot(df,column):
    data =df[[column]] 
    encoder = OneHotEncoder(sparse_output =False)
    coded = encoder.fit_transform(data)
    column_names = encoder.get_feature_names_out([column])
    new_df = pd.DataFrame(coded, columns=column_names)
    new_df.index = df.index
    df = pd.concat([df, new_df], axis=1)
    df = df.drop(columns=column)
    return df

def boxplot(target):
    sns.boxplot(target)
    plt.title("Visualizando distribuição do target")
    plt.savefig('D:/trabalhos pos/Aprendizado Supervisionado Entregas/5. Regressao Linear/distribuicao.png')
    plt.clf

def histograma(target):
    plt.hist(target, bins=30)
    plt.title('Histograma do target')
    plt.savefig('D:/trabalhos pos/Aprendizado Supervisionado Entregas/5. Regressao Linear/histograma.png')
    plt.clf

def iqr(df,column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR

    outliers = df[(df[column] < limite_inferior) | (df[column] > limite_superior)]
    return outliers

def min_max(df,column):
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])
    return df

df = read_csv('D:/trabalhos pos/Aprendizado Supervisionado Entregas/5. Regressao Linear/clientes_regressao_linear.csv')
df = number_pattern(df,'renda_mensal')
df = number_pattern(df,'avaliacao_cliente')
df = number_pattern(df,'bonus_fidelidade')
df = one_hot(df,'profissao')
df = min_max(df,'renda_mensal')
df = min_max(df,'renda_mensal')
save_to_csv(df,'D:/trabalhos pos/Aprendizado Supervisionado Entregas/5. Regressao Linear/corrigido.csv' )
X = df.drop(['id','valor_proxima_compra'], axis=1)
y = df['valor_proxima_compra']

x_train,x_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
boxplot(y)
histograma(y)
outliers = iqr(df,'valor_proxima_compra')
save_to_csv(outliers,'D:/trabalhos pos/Aprendizado Supervisionado Entregas/5. Regressao Linear/outliers.csv')
