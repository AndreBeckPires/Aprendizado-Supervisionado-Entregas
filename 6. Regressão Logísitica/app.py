import pandas as pd
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import seaborn as sns
import matplotlib.pyplot as plt
def read_csv(path):
    df = pd.read_csv(path)

    #print(df.head())
    return df

def save_to_csv(df,path):
    df.to_csv(path, index = False)

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



def number_pattern(df,column):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df[column].fillna(df[column].median(), inplace = True)
    return df
def min_max(df,column):
    scaler = MinMaxScaler()
    df[column] = scaler.fit_transform(df[[column]])
    return df

#limpeza
df = read_csv('clientes_logistica.csv')
df = one_hot(df,'profissao')
df = one_hot(df,'canal_ultima_compra')
df = min_max(df,'renda_mensal')
save_to_csv(df,'corrigido.csv')

#separando dados
X = df.drop(['id','comprou_novamente'],axis=1)
save_to_csv(X,'x.csv')
y = df['comprou_novamente']
save_to_csv(y,'y.csv')
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=42)
model = LogisticRegression()
model.fit(x_train,y_train)


#teste
y_pred = model.predict(x_test)
y_prob = model.predict_proba(x_test)[:, 1]

limiar = 0.3
y_pred_novo = (y_prob >= limiar).astype(int)

#acuracia
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#Matriz de confusão
conf = confusion_matrix(y_test, y_pred)
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão")
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.savefig('confusao_limiar_padrao.png')
plt.clf()

#Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.savefig('roc_limiar_padrao.png')
plt.clf()


#Matriz de confusão
conf = confusion_matrix(y_test, y_pred_novo)
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão")
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.savefig('confusao_limiar_novo.png')
plt.clf()
