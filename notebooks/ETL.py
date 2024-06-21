from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
# --------------------------------------------------------------------- # 
from datetime import datetime, timedelta
import re




# Transformar dados inconsistentes para NaN
class TransformToNull(BaseEstimator, TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
         
        for column_name in self.column_names:
            if pd.api.types.is_object_dtype(X_transformed[column_name]):
                #Preenchendo valores nulos para vazios para previnir erros.
                
                X_transformed[column_name] = X_transformed[column_name].fillna('')
                # Verificando se contém apenas caracteres especiais.
                X_transformed.loc[(X_transformed[column_name].str.match(r'[()\-$#@!%&*]', na=False)) & ~(X_transformed[column_name].str.contains(r'[^0-9]' ,regex=True)), column_name] = np.nan
                # #Verificando se possui o valor NM (Not Mentioned).
                X_transformed.loc[X_transformed[column_name].str.contains("NM", na=False), column_name] = np.nan
                #Verificando valores vazios e transformando em nulos.
                X_transformed.loc[X_transformed[column_name].str.strip().eq(''), column_name] = np.nan
                
        return X_transformed


# Realizando tratamento dos dados nulos da coluna Num_Credit_Card.
class CleaningMissingCreditCard:
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        X_transformed.loc[X_transformed[self.column_name] <= 0, self.column_name] = 1
                
        return X_transformed


# Realizando tratamento dos dados nulos da coluna Type_of_Loan.
class CleaningMissingTypeOfLoan:
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        X_transformed[self.column_name].fillna("Not Specified", inplace=True)
                
        return X_transformed
    

# Realizando tratamento dos dados nulos da coluna Num_of_Delayed_Payment.
class CleaningMissingDelayedPayment:
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        X_transformed[self.column_name].fillna(0, inplace=True)
        mode_num_delayed_payment = X_transformed.groupby("Customer_ID")[self.column_name].apply(lambda x: x.mode().iloc[0])
        X_transformed.loc[X_transformed[self.column_name] == 0, self.column_name] = X_transformed["Customer_ID"].map(mode_num_delayed_payment)
        X_transformed.loc[(X_transformed[self.column_name] == 0) & X_transformed['Delay_from_due_date'] > 0, self.column_name] = 1
                
        return X_transformed


# Realizando tratamento dos dados nulos da coluna Monthly_Inhand_Salary.
class CleaningMissingMonthlySalary:
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        # pegando os customer_id que possuem NaN na coluna
        monthly_salaray_nan = X_transformed[X_transformed[self.column_name].isna()]['Customer_ID'].unique()  

        # buscando apenas os customer_id que estejam na variavel para verificar se ele possui algum valor em outro mês   
        customers_id = X_transformed[X_transformed['Customer_ID'].isin(monthly_salaray_nan)]

        #colocando a media para cada Customer_id na coluna de salario
        mean_salary  = customers_id.groupby("Customer_ID")[self.column_name].apply(lambda x:  x.median())
        mean_salary = pd.DataFrame(mean_salary) # transformando em um df

        X_transformed.loc[X_transformed[self.column_name].isna(), self.column_name] = customers_id["Customer_ID"].map(mean_salary[self.column_name])
        
                
        return X_transformed
    

# Realizando tratamento dos dados nulos da coluna Num_Bank_Accounts.
class CleaningNumBankAccounts:
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        # agrupando pelo costumerID e verificando quais possuem valores nulos
        num_bank_account_null=  X_transformed.groupby('Customer_ID').apply(lambda x: x[(x[self.column_name] <= 0)]) 
        index_bank_account_null = num_bank_account_null.droplevel(-1).index #pegando os customer_id

        # pegando apenas os dados que possuem o customer_id contido na variavel num_bank_account_null
        filtered_df = X_transformed[X_transformed['Customer_ID'].isin(index_bank_account_null)] 
        count_per_customer_id = filtered_df['Customer_ID'].value_counts() #pegar a quantidade de customer_id para realizar o replace

        X_transformed.loc[X_transformed['Customer_ID'].isin(index_bank_account_null) & (X_transformed[self.column_name] <= 0), self.column_name] = X_transformed['Customer_ID'].map(count_per_customer_id) 

        return X_transformed


# Realizando tratamento dos dados nulos.
class CleaningMissingValues:
    def __init__(self, column_names):
        self.column_names = column_names
    
    def fit(self, X, y=None):
        return self   

    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        for column_name in self.column_names:
            #Outras colunas serão tratadas dessa forma.
            X_transformed[column_name] = X_transformed.groupby('Customer_ID')[column_name].transform(lambda x: x.bfill().ffill())

        return X_transformed


# Retirando caracteres não numéricos de colunas "numéricas"
class CleaningNotNumbers(BaseEstimator, TransformerMixin): 
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        for column_name in self.column_names:
            # Verificando se a coluna é do tipo Object.
            if pd.api.types.is_object_dtype(X_transformed[column_name]):
                if X_transformed[column_name].str.contains(r'\d', regex=True).any():  # Verificando se contém algum valor numérico.
                    # Realizando a limpeza de caracteres inválidos.
                    X_transformed[column_name] = X_transformed[column_name].str.replace(r'[^0-9]', '', regex=True)
                    # Converter a coluna para numérico
                    X_transformed[column_name] = pd.to_numeric(X_transformed[column_name], errors='coerce')
            else:
                # Se é do tipo float ou int
                X_transformed[column_name] = X_transformed[column_name].abs()
                
        return X_transformed
    
class ModifyMonthCreditHistory(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        return self   
    
    def modify_month(self, df_group):
        for i in range(len(df_group)):
            value = df_group.iloc[i][self.column_name]
            # Verifica se o valor é uma string antes de tentar dividi-lo
            if isinstance(value, str):
                parts = value.split(' and ')
                if len(parts) > 1:
                    months_part = parts[1].split()
                    months_part[0] = str(i + 1)  # Adicionando o valor para o mês com o índice mais 1.
                    df_group.at[df_group.index[i], self.column_name] = f"{parts[0]} and {' '.join(months_part)}"
        return df_group

    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        # chamando a função criada para realizar as alterações
        X_transformed[self.column_name] = X_transformed.groupby('Customer_ID').apply(self.modify_month).reset_index(drop=True) 

        return X_transformed


class CreateDateCreditHistoryColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self  
    

    # fazer alteração da string para uma data
    def convert_to_datetime(self, duration_str):
        #realizando a divisão em 2 grupos, o primeiro com o valor do ano e o segundo com o valor do Mês
        match = re.match(r'(\d+) Years and (\d+) Months', duration_str)
        if match:
            years = int(match.group(1))
            months = int(match.group(2))
        
        
        base_date = datetime.now()
        
        # Realizando a diferença entre a data atual e a data de ingressão do usuário.
        new_date = base_date - timedelta(days=years*365 + months*30)
        
        # Retornar a data no formato YYYY-MM
        return new_date.strftime('%Y-%m')
    
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        # aplicando a função e criando uma nova coluna no dataframe.
        X_transformed['Credit_History_Age_Date'] = X_transformed[self.column_name].apply(self.convert_to_datetime) 
                
        return X_transformed
    

class CreateMonthNumberColumn(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self  
        
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        month_dic = { 'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                        'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12 }
        
        # Criação da coluna para indentificar os meses em números
        X_transformed["Number_Month"] = X_transformed[self.column_name].str.lower().map(month_dic) 
                
        return X_transformed
    

class TransformToBinaryValues(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        

        X_transformed[self.column_name] = X_transformed[self.column_name].map({'Yes': 1, 'No': 0}) # transformando os dados para binário
                
        return X_transformed


class ConvertDtypeToNumeric(BaseEstimator, TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        for column_name in self.column_names:
            if X_transformed[column_name].str.contains(r'\d', regex=True).any():
                X_transformed[column_name] = pd.to_numeric(X_transformed[column_name], errors='coerce')
            else:
                pass
                
        return X_transformed
    
# tratar outliers!!! 





