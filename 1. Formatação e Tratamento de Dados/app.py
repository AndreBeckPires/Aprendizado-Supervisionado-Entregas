import pandas as pd
from collections import Counter

def read_csv(path):
    df = pd.read_csv(path)

    #print(df.head())
    return df

def string_pattern(df,column):
    df[column] = df[column].str.strip().str.title()
    return df

def email(df):
    df = df.dropna(subset=['email'])
    df['email'] = df['email'].apply(lambda x : x + '.com' if x and not x.endswith('.com') and not x.endswith('@') else x)
    unique = df['email'].unique()
    domains = []
    for email in unique:
        domain = email.split('@')
        if(len(domain)>1):
            domains.append(domain[1])
    domains = Counter(domains)
    most_common = domains.most_common(1)
    print(most_common[0][0])
    df['email'] = df['email'].apply(lambda x : x + f'{most_common[0][0]}' if x and  x.endswith('@') else x)
    return df

def save_to_csv(df,path):
    df.to_csv(path)

def number_pattern(df,column):
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df[column].fillna(df[column].median(), inplace = True)
    return df


df = read_csv('1. Formatação e Tratamento de Dados\dataset_clientes.csv')
df = string_pattern(df,'nome')
df = string_pattern(df,'cidade')
df = string_pattern(df,'ativo')
df = email(df)
df = number_pattern(df,'idade')
df = number_pattern(df,'compras')
df = df.dropna()
save_to_csv(df,'1. Formatação e Tratamento de Dados\corrigido.csv')
#print(df.head())