import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import pickle

# 1. Carregar os dados
print("Carregando dados...")
df = pd.read_csv('scredito_clientes.csv', sep=';')

# 2. Pré-processamento
print("Pré-processando dados...")

# Converter colunas numéricas que estão como string (com vírgula)
cols_to_fix = ['renda_familiar']
for col in cols_to_fix:
    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# Criar a variável alvo (Target) baseada no Score de Crédito
# Regra de Negócio Simplificada:
# Score < 500: Risco Alto (0) - Negado
# Score >= 500: Risco Baixo (1) - Aprovado
df['target'] = df['score_credito'].apply(lambda x: 1 if x >= 500 else 0)

# Selecionar Features (Colunas que vamos usar para prever)
features = ['idade', 'renda_familiar', 'regiao', 'escolaridade', 'tem_negocio_proprio']
X = df[features]
y = df['target']

# Converter variáveis categóricas (texto) em números
le_regiao = LabelEncoder()
X.loc[:, 'regiao'] = le_regiao.fit_transform(X['regiao'])

le_escolaridade = LabelEncoder()
X.loc[:, 'escolaridade'] = le_escolaridade.fit_transform(X['escolaridade'])

# Tratamento para booleanos (True/False)
X.loc[:, 'tem_negocio_proprio'] = X['tem_negocio_proprio'].astype(int)

# 3. Divisão Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Treinamento do Modelo
print("Treinando modelo Random Forest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 5. Avaliação
y_pred = clf.predict(X_test)
print("\n--- Relatório de Performance ---")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}")
print(classification_report(y_test, y_pred))

# 6. Salvar o Modelo e os Encoders
print("Salvando modelo...")
model_data = {
    'model': clf,
    'le_regiao': le_regiao,
    'le_escolaridade': le_escolaridade
}

with open('modelo_credito.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Concluído! Modelo salvo em 'modelo_credito.pkl'")
