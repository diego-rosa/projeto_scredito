from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)  # Habilitar CORS para permitir chamadas do frontend

# Carregar o modelo e os encoders
print("Carregando modelo...")
with open('modelo_credito.pkl', 'rb') as f:
    model_data = pickle.load(f)

clf = model_data['model']
le_regiao = model_data['le_regiao']
le_escolaridade = model_data['le_escolaridade']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extrair dados do JSON
        # Esperamos: { 'idade': 30, 'renda_familiar': 2000.0, 'regiao': 'Zona Norte', 'escolaridade': 'Médio', 'tem_negocio_proprio': True }
        
        # Preparar o DataFrame para o modelo
        input_data = pd.DataFrame([{
            'idade': int(data['idade']),
            'renda_familiar': float(data['renda_familiar']),
            'regiao': data['regiao'],
            'escolaridade': data['escolaridade'],
            'tem_negocio_proprio': 1 if data['tem_negocio_proprio'] else 0
        }])
        
        # Aplicar os mesmos encoders do treinamento
        # Nota: Em produção real, precisaríamos tratar valores não vistos (unknown labels)
        try:
            input_data['regiao'] = le_regiao.transform(input_data['regiao'])
        except ValueError:
            # Fallback para um valor padrão se a região não existir no treino
            input_data['regiao'] = le_regiao.transform([le_regiao.classes_[0]])[0]
            
        try:
            input_data['escolaridade'] = le_escolaridade.transform(input_data['escolaridade'])
        except ValueError:
            input_data['escolaridade'] = le_escolaridade.transform([le_escolaridade.classes_[0]])[0]

        # Fazer a predição
        prediction = clf.predict(input_data)[0]
        probability = clf.predict_proba(input_data)[0][1] # Probabilidade de ser classe 1 (Aprovado)
        
        # Regra de resposta
        if prediction == 1:
            status = "APROVADO"
            mensagem = "Parabéns! Seu crédito foi pré-aprovado. Nossa equipe entrará em contato em breve."
            cor = "#00B27A" # Verde
        else:
            status = "EM ANÁLISE"
            mensagem = "No momento, não conseguimos aprovar o valor total. Que tal uma mentoria financeira gratuita?"
            cor = "#FFB400" # Amarelo (ou Vermelho #FF5A5F)

        return jsonify({
            'status': status,
            'mensagem': mensagem,
            'score_interno': float(probability),
            'cor': cor
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
