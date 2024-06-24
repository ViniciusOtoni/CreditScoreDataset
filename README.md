# Credit Score Dataset Transformation - AUTOR: Vinícius Otoni da Silva

Este projeto contém transformações de dados aplicadas ao conjunto de dados de pontuação de crédito. As transformações foram desenvolvidas para limpar e preparar os dados para análises subsequentes.

## Descrição

O repositório inclui classes para transformar colunas específicas de um DataFrame, aplicando regras de transformação para valores nulos, outliers e outras condições específicas.


## Extrutura do projeto

data/: Contém o dataset de creditscore utilizado na análise. 
<br>
notebooks/: Notebooks Jupyter com o processo de exploração e análise de dados.
<br> 
---n00_data_visualization Realizar visualização dos dados e extrutura do dataframe.
<br>
---n01_data_cleaning Processo de (data cleaning) do dataframe.
<br> 
---n02_pipeline Criação do pipeline para automatizar o processo de transformações do dataframe.
<br> 
---ETL.py Criação das classes com os métodos fit e transform responsável pelo processo de ETL ( extract | transform | load ) do dataframe.


### Classes e Métodos

#### `CleaningNumBankAccounts`

Esta classe transforma valores nulos agrupados pela coluna `Customer_ID` e preenche os valores pela quantidade de ocorrências de cada `Customer_ID`.

- **Parâmetros:**
  - `column_name` (str): Nome da coluna no DataFrame a ser transformada.

- **Métodos:**
  - `fit(X, y=None)`: Método utilizado para conformidade com o pipeline do Scikit-Learn. Não realiza nenhuma ação.
  - `transform(X)`: Aplica a transformação na coluna especificada.

### Exemplo de Uso

```python
import ETL
import pandas as pd

# Criando um DataFrame de exemplo
data = {
    'Customer_ID': [1, 1, 2, 2, 3, 3],
    'Num_Bank_Accounts': [0, 1, 0, 2, 0, 0]
}
df = pd.DataFrame(data)



# Aplicando a transformação
transformer = ETL.CleaningNumBankAccounts(column_name='Num_Bank_Accounts')
df_transformed = transformer.fit_transform(df)

print(df_transformed)

```



