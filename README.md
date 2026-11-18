# ML Lab — Laboratório de Machine Learning

Aplicação educacional de Machine Learning desenvolvida em **Python + Streamlit** para executar, de forma visual, o fluxo completo de um projeto supervisionado de regressão ou classificação.

O ML Lab permite importar e explorar dados, configurar o pré-processamento, treinar diferentes algoritmos, avaliar métricas, realizar novas predições e exportar os resultados e modelos treinados.

## Fluxo da aplicação

**Importar dados → Explorar → Pré-processar → Selecionar modelo → Treinar → Avaliar → Predizer → Exportar**

## Tecnologias

- Python 3.11+
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Joblib

## Funcionalidades

### Importação e exploração

- Importação de arquivos CSV, TXT e JSON.
- Configuração de separador para arquivos TXT.
- Datasets demonstrativos para experimentação.
- Identificação de linhas, colunas e tipos de dados.
- Análise de valores ausentes.
- Seleção da variável-alvo e das features.
- Estatísticas descritivas.
- Histogramas e visualizações exploratórias.
- Matriz de correlação.

### Pré-processamento

O tratamento dos dados é integrado ao treinamento por meio de `Pipeline` e `ColumnTransformer`, ajudando a evitar data leakage.

- `SimpleImputer` para valores ausentes.
- `StandardScaler`.
- `MinMaxScaler`.
- `OneHotEncoder` para atributos categóricos.
- Separação entre treino e teste.
- Divisão estratificada quando aplicável.
- Validação cruzada.
- `random_state=42` para reprodutibilidade.

## Modelos disponíveis

### Regressão

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Ridge
- Lasso

### Classificação

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)

A interface permite configurar hiperparâmetros e comparar o desempenho de diferentes modelos.

## Avaliação

### Regressão

- RMSE
- MAE
- MSE
- R²
- Comparação entre valores reais e previstos
- Análise de desempenho em treino e teste

### Classificação

- Acurácia
- Precisão
- Recall
- F1-score
- Matriz de confusão
- Relatório de classificação
- Probabilidades das classes quando suportadas pelo modelo

## Predições

Depois do treinamento, o ML Lab permite usar o pipeline treinado para novos dados.

- Entrada manual dos atributos.
- Predição em lote por arquivo.
- Resultado numérico para regressão.
- Classe prevista para classificação.
- Probabilidades quando disponíveis.
- Exportação em CSV, JSON e TXT.
- Salvamento e carregamento de modelos com Joblib.

## Módulo educacional

A aplicação também apresenta explicações sobre conceitos utilizados durante os experimentos.

- Aprendizado supervisionado.
- Regressão e classificação.
- Parâmetros e hiperparâmetros.
- Função de custo e métricas.
- Gradiente descendente e taxa de aprendizado.
- Overfitting e underfitting.
- Trade-off viés-variância.
- Regularização L1 e L2.
- Árvores de decisão e CART.
- Gini e entropia.
- Bagging e bootstrap.
- Out-of-Bag (OOB).

## Estrutura do projeto

```text
ML-LAB/
├── app.py
├── requirements.txt
├── README.md
├── ml_lab/
│   ├── __init__.py
│   ├── data.py
│   ├── modeling.py
│   ├── evaluation.py
│   └── persistence.py
├── examples/
│   ├── satisfacao.csv
│   ├── classificacao.csv
│   ├── registros_predicao.json
│   └── teste_ml_lab.csv
└── tests/
    └── test_core.py
```

## Como executar

### 1. Entre na pasta do projeto

```powershell
cd "C:\Projetos\ML-LAB"
```

### 2. Crie um ambiente virtual

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Instale as dependências

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Inicie a aplicação

```powershell
python -m streamlit run app.py
```

O Streamlit normalmente abrirá a aplicação automaticamente no navegador. O endereço local padrão é `http://localhost:8501`.

## Dataset para teste rápido

O arquivo `examples/teste_ml_lab.csv` pode ser usado para testar classificação com dados numéricos e categóricos.

Configuração sugerida:

- Target: `comprou`
- Problema: classificação
- Features: todas as demais colunas
- Modelos: Logistic Regression ou Random Forest Classifier

Também estão disponíveis exemplos para satisfação e classificação, além de registros JSON para testar novas predições.

## Testes automatizados

Execute:

```powershell
python -m pytest -q
```

Os testes cobrem funcionalidades centrais dos módulos do laboratório.

## Objetivo acadêmico

O projeto foi construído para permitir que conceitos de Machine Learning sejam observados durante a execução de experimentos reais. Os resultados exibidos são produzidos por modelos efetivamente treinados com Scikit-learn.

O foco é facilitar a compreensão prática da preparação dos dados, treinamento, generalização, avaliação e uso de modelos supervisionados.
