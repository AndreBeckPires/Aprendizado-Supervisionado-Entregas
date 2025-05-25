import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from category_encoders import HashingEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from category_encoders import HashingEncoder

def read_csv(path):
    df = pd.read_csv(path)

    #print(df.head())
    return df

def save_to_csv(df,path):
    df.to_csv(path, index = False)



def string_pattern(df,column):
    df[column] = df[column].str.strip().str.title()
    return df

def number_pattern(df,column):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df[column].fillna(df[column].median(), inplace = True)
    return df

def date(df,column):
    df[column] = pd.to_datetime(df[column], errors = 'coerce')
    return df

def fix_sim_nao(df,column):
    df[column] = df[column].apply(lambda x : 'Sim' if x and  x == 'S' else x)
    df[column] = df[column].apply(lambda x : 'Não' if x and  x == 'N' else x)
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

def label(df,columns):
    label_encoder = LabelEncoder()
    for col in columns:
        df[f'{col}_encoded'] = label_encoder.fit_transform(df[col])
    return df

def target(df,categorico,numerico):
    x = df[categorico]
    y = df[numerico]
    encoder = TargetEncoder()
    x_encoded = encoder.fit_transform(x,y)

    df[f'{categorico}/{numerico}'] = x_encoded
    return df

def hash_encod(df,column):
    encoder = HashingEncoder(n_components= 8, cols=[column])
    df_encoded = encoder.fit_transform(df)
    return df_encoded


df = read_csv('4. Codificação de Formatação de Dataset e Ajuste de Variáveis Categóricas/animais_abrigo.csv')
df = string_pattern(df,'nome')
df = string_pattern(df,'especie')
df = string_pattern(df,'tamanho')
df = string_pattern(df,'cor')
df = number_pattern(df,'idade')
df = date(df,'data_resgate')
df = df.dropna()
df = string_pattern(df,'adotado')
df = fix_sim_nao(df,'adotado')
df = one_hot(df,'especie')
df = label(df,['tamanho','cor','adotado'])
df = target(df,'adotado_encoded','idade')
df = target(df,'adotado_encoded','tamanho_encoded')
df = hash_encod(df,'local_resgate')

df = df.drop(columns = [
    'tamanho',
    'cor',
    'adotado'
])
print(df.head())
save_to_csv(df,'4. Codificação de Formatação de Dataset e Ajuste de Variáveis Categóricas/corrigido.csv')