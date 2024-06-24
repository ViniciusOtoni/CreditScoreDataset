from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
# --------------------------------------------------------------------- # 
from datetime import datetime, timedelta
import re
from math import ceil




# Transformar dados inconsistentes para NaN
class TransformToNull(BaseEstimator, TransformerMixin):
    """
    Transforma valores irregulares em nulos.
    
    Parâmetros:
    columns: list of str
        Nomes das colunas no DataFrame a serem transformadas.

    Métodos:
    fit: Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
    transform: Aplica a transformação para valores nulos nas colunas especificadas.
    """

    def __init__(self, column_names):
        self.column_names = column_names
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.

        for column_name in self.column_names:
            if pd.api.types.is_object_dtype(X_transformed[column_name]):
                # Preenchendo valores nulos com vazios para prevenir erros.
                X_transformed[column_name] = X_transformed[column_name].fillna('')
               
                # # Verificando se contém caracteres especiais para SSN.
                if column_name == 'SSN' or column_name == 'Payment_Behaviour':
                    special_chars_mask = X_transformed[column_name].str.contains(r'[()\$#@!%&*]', na=False)
                else:
                    # Verificando se contém apenas caracteres especiais ou caracteres especiais com números.
                    special_chars_mask = X_transformed[column_name].str.contains(r'[()\-_$#@!%&*]', na=False) & ~X_transformed[column_name].str.contains(r'[a-zA-Z0-9]', na=False)

                X_transformed.loc[special_chars_mask, column_name] = np.nan

                # Verificando se possui o valor NM (Not Mentioned).
                X_transformed.loc[X_transformed[column_name].str.contains("NM", na=False), column_name] = np.nan

                # Verificando valores vazios e transformando em nulos.
                X_transformed.loc[X_transformed[column_name].str.strip().eq(''), column_name] = pd.NA

        return X_transformed

# Realizando tratamento dos dados nulos da coluna Num_Credit_Card.
class CleaningMissingCreditCard(BaseEstimator, TransformerMixin):
    
    """
    Transforma valores menores ou igual a 0 para 1.
    
    Parâmetros:
    column: Num_Credit_Card

    Métodos:
    fit: Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
    transform: Aplica a transformação na coluna especificada.
    """

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        X_transformed.loc[X_transformed[self.column_name] <= 0, self.column_name] = 1
                
        return X_transformed


# Realizando tratamento dos dados nulos da coluna Type_of_Loan.
class CleaningMissingTypeOfLoan(BaseEstimator, TransformerMixin):

    """
    Transforma valores nulos para a str (Not Specified).
    
    Parâmetros:
    column: Type_of_Loan

    Métodos:
    fit:  Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
    transform: Aplica a transformação na coluna especificada.
    """

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        X_transformed[self.column_name].fillna("Not Specified", inplace=True)
                
        return X_transformed
    

# Realizando tratamento dos dados nulos da coluna Num_of_Delayed_Payment.
class CleaningMissingDelayedPayment(BaseEstimator, TransformerMixin):

    """
    Transforma valores nulos agrupados pela coluna Customer_ID e preenchendo os valores pela moda de cada Customer_ID.
    
    Parâmetros:
    column: Num_of_Delayed_Payment

    Métodos:
    fit:  Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
    transform: Aplica a transformação na coluna especificada.
    """

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
class CleaningMissingMonthlySalary(BaseEstimator, TransformerMixin):

    """
    Transforma valores nulos agrupados pela coluna Customer_ID e preenchendo os valores pela media de cada Customer_ID.
    
    Parâmetros:
    column: Monthly_Inhand_Salary

    Métodos:
    fit:  Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
    transform: Aplica a transformação na coluna especificada.

    """

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
class CleaningNumBankAccounts(BaseEstimator, TransformerMixin):

    """
    Transforma valores nulos agrupados pela coluna Customer_ID e preenche os valores pela quantidade de ocorrências de cada Customer_ID.
    
    Parâmetros:
    column: Num_Bank_Accounts

    Métodos:
    fit:  Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
    transform: Aplica a transformação na coluna especificada.

    """

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



# Realizando tratamento dos dados nulos da coluna Monthly_Balance.
class CleaningMissingMonthlyBalance(BaseEstimator, TransformerMixin):

    """
    Transforma valores nulos da coluna especificada para a moda da coluna.
    
    Parâmetros:
    column: Monthly_Balance

    Métodos:
    fit:  Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
    transform: Aplica a transformação na coluna especificada.

    """

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        X_transformed.loc[X_transformed[self.column_name].isna(), self.column_name] = X_transformed[self.column_name].mode().iloc[0]
                
        return X_transformed


# Realizando tratamento dos dados nulos.
class CleaningMissingValues(BaseEstimator, TransformerMixin):

    """
    Transforma valores nulos pela (próxima ou anterior) ocorrência de valor agrupado pelo Customer_ID.
    
    Parâmetros:
    columns: list of str
        Nomes das colunas no DataFrame a serem transformadas.

    Métodos:
    fit: Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
    transform: Aplica a transformação para valores nulos nas colunas especificadas.

    """
    
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

    """
    Retira valores inconsistentes de colunas "númericas" e realiza a conversão do dtype.
    
    Parâmetros:
    columns: list of str
        Nomes das colunas no DataFrame a serem transformadas.

    Métodos:
    fit: Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
    transform: Aplica a transformação para valores nulos nas colunas especificadas.
    
    """

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

# Ajustando os meses contidos na string.    
class ModifyMonthCreditHistory(BaseEstimator, TransformerMixin):

    """
    Substitui os meses adequando a sequência correta.
    
    Parâmetros:
    column: Credit_History_Age

    Métodos:
    fit: Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
    transform: Aplica a transformação para valores nulos nas colunas especificadas.
    
    """
     

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
        X_transformed = X_transformed.groupby('Customer_ID').apply(self.modify_month).reset_index(drop=True) 

        return X_transformed


# Criando uma nova coluna no dataframe contendo apenas o ano e mês.
class CreateDateCreditHistoryColumn(BaseEstimator, TransformerMixin):

    """
    Realiza a criação de uma nova coluna (Credit_History_Age_Date) contendo apenas o mês e ano respctivos da coluna (Credit_History_Age).
    
    Parâmetros:
    column: Credit_History_Age

    Métodos:
    fit: Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
    transform: Aplica a transformação para valores nulos nas colunas especificadas.
    
    """


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
    

# Criando uma coluna númerica para os meses
class CreateMonthNumberColumn(BaseEstimator, TransformerMixin):

    """
    Realiza a criação de uma nova coluna (Credit_History_Age_Date) contendo apenas o mês e ano respctivos da coluna (Credit_History_Age).
    
    Parâmetros:
    column: Credit_History_Age

    Métodos:
    fit: Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
    transform: Aplica a transformação para valores nulos nas colunas especificadas.
    
    """


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
    

# Alterando a coluna para binário (yes: 1) (no: 0)
class TransformToBinaryValues(BaseEstimator, TransformerMixin):

    """
    Realiza a conversão da coluna (Payment_of_Min_Amount) para valor binário { Yes: 1, No: 0 }.
    
    Parâmetros:
    column: Payment_of_Min_Amount

    Métodos:
    fit: Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
    transform: Aplica a transformação para valores nulos nas colunas especificadas.
    
    """

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        

        X_transformed[self.column_name] = X_transformed[self.column_name].map({'Yes': 1, 'No': 0}) # transformando os dados para binário
                
        return X_transformed


# Convertendo as colunas possíveis para o dtype int ou float.
class ConvertDtypeToNumeric(BaseEstimator, TransformerMixin):

    """
    Realiza a conversão da colunas específicadas para o dtype numérico.
    
    Parâmetros:
    columns: list of str
        Nomes das colunas no DataFrame a serem transformadas.

    Métodos:
    fit: Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
    transform: Aplica a transformação para valores nulos nas colunas especificadas.
    
    """
     

    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        for column_name in self.column_names:
            if pd.api.types.is_object_dtype(X_transformed[column_name]):
                if X_transformed[column_name].str.contains(r'\d', regex=True).any():
                    X_transformed[column_name] = pd.to_numeric(X_transformed[column_name], errors='coerce')
            else:
                pass
                
        return X_transformed
    

# CLASSES DE TRATAMENTO DE OUTLIERS.


class TreatingOutliersWithQuantile(BaseEstimator, TransformerMixin):

    """
    Realiza o tratamento de outliers através de um limitador gerado pelo quartill/quadrante.
    
    Parâmetros:
    columns: list of float/int
        Nomes das colunas no DataFrame a serem transformadas.

    Métodos:
    fit: Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
    transform: Aplica a transformação para valores nulos nas colunas especificadas.
    
    """

    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        for column_name in self.column_names:
            Q1 = X_transformed[column_name].quantile(0.25) # primeiro quartil (pega os valores de 25% para baixo)
            Q3 = X_transformed[column_name].quantile(0.95) # terceiro quartil (pega os valores de 95% para baixo)

            if(column_name == 'Outstanding_Debt' or column_name == 'Amount_invested_monthly'):
                Q3 = X_transformed[column_name].quantile(0.75) # terceiro quartil (pega os valores de 75% para baixo)
           
            
            IQR = Q3 - Q1 # calcula a diferença entre o primeiro e o terceiro quartil

            upper_bound = Q3 + 1.5 * IQR

            X_transformed[column_name] = X_transformed[column_name].apply(lambda x:  ceil(upper_bound) if x > upper_bound else x) # Alteração de outliers
                
        return X_transformed



class TreatingOutliersWithMode(BaseEstimator, TransformerMixin):
    
    """
    Realiza o tratamento de outliers aplicando a moda da coluna.
    
    Parâmetros:
    columns: list of float/int
        Nomes das colunas no DataFrame a serem transformadas.

    Métodos:
    fit: Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
    transform: Aplica a transformação para valores nulos nas colunas especificadas.
    
    """

    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        for column_name in self.column_names:
            limit = 100
            if column_name == 'Interest_Rate':
                limit = 30
            X_transformed.loc[X_transformed[column_name] >= limit, column_name] =  X_transformed[column_name].mode().iloc[0] # Alteração de outliers
                
        return X_transformed

class TreatingOutliersNumCreditInquires(BaseEstimator, TransformerMixin): #Num_Credit_Inquiries

    """
    Realiza o tratamento de outliers aplicando a moda da coluna agrupado pelo Customer_ID.
    
    Parâmetros:
    column: Num_Credit_Inquiries
        Nomes da coluna no DataFrame a serem transformadas.

    Métodos:
    fit: Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
    transform: Aplica a transformação para valores nulos nas colunas especificadas.
    
    """

    def __init__(self, column_name):
        self.column_name = column_name

    def fit(self, X, y=None):
        return self  
    
    def transform(self, X):
        X_transformed = X.copy()  # Copiando o input do dataframe para evitar modificar o original.
        
        
        X_transformed[self.column_name] = X_transformed.groupby('Customer_ID')[self.column_name].transform(
            lambda x: x.mode().iloc[1] if len(x.mode()) > 1 else x.mode().iloc[0] if x.mode().iloc[0] <= 20 else x)
                
        return X_transformed




