import pandas as pd
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

df = read_csv('3. Codificação de Variáveis Categóricas\clientes_vendas.csv')

df = one_hot(df,'categoria_produto')

df = label(df,['genero','estado_civil','canal_venda'])

df = target(df,'cidade','avaliacao_cliente')

df = hash_encod(df,'produto')
save_to_csv(df,'3. Codificação de Variáveis Categóricas\corrigido.csv')




# Hash Encoding: Mapeamento usando funções hash (útil para muitas categorias). PROFISSAO e PRODUTO