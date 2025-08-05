from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
import os

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# 전역 변수로 모델 저장
models = {
    'lstm_model': None,
    'rf_models': None,
    'reg_models': None,
    'last_lotto_numbers': None
}

def load_and_prepare_data(file_path='lotto.xlsx'):
    """엑셀 파일에서 로또 데이터를 불러오고 모델 학습에 맞게 전처리합니다."""
    try:
        df = pd.read_excel(file_path)
        df = df.sort_values('회차')
        numbers = df[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']].values
    except Exception as e:
        print(f"데이터 로드 오류: {str(e)}")
        return None, None, None

    X, y = [], []
    for i in range(len(numbers) - 1):
        X.append(numbers[i])
        y.append(numbers[i+1])
        
    return np.array(X), np.array(y), numbers[-1]

def train_models(X, y):
    """세 가지 다른 모델을 학습시킵니다."""
    
    # LSTM 모델
    X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
    lstm_model = Sequential([
        LSTM(128, activation='relu', input_shape=(X_lstm.shape[1], X_lstm.shape[2]), return_sequences=True),
        LSTM(64, activation='relu'),
        Dense(6)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_lstm, y, epochs=50, batch_size=10, verbose=0)

    # 랜덤 포레스트 모델
    rf_models = []
    for i in range(y.shape[1]):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y[:, i])
        rf_models.append(model)
    
    # 로지스틱 회귀 모델
    reg_models = []
    for i in range(y.shape[1]):
        model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
        model.fit(X, y[:, i])
        reg_models.append(model)
    
    return lstm_model, rf_models, reg_models

def predict_next_lotto(last_numbers, lstm_model, rf_models, reg_models):
    """학습된 모델들로 다음 회차 로또 번호를 예측합니다."""
    
    last_numbers_reshaped = last_numbers.reshape(1, 6)
    last_numbers_lstm = last_numbers.reshape(1, 1, 6)
    
    # LSTM 예측
    lstm_pred = lstm_model.predict(last_numbers_lstm, verbose=0)[0]
    lstm_pred = np.clip(np.round(lstm_pred), 1, 45).astype(int)
    lstm_result = sorted(list(set(lstm_pred)))
    while len(lstm_result) < 6:
        new_num = np.random.randint(1, 46)
        if new_num not in lstm_result:
            lstm_result.append(new_num)
    lstm_result = sorted(lstm_result[:6])

    # 랜덤 포레스트 예측
    rf_pred = [model.predict(last_numbers_reshaped)[0] for model in rf_models]
    rf_pred = [max(1, min(45, int(pred))) for pred in rf_pred]
    rf_result = sorted(list(set(rf_pred)))
    while len(rf_result) < 6:
        new_num = np.random.randint(1, 46)
        if new_num not in rf_result:
            rf_result.append(new_num)
    rf_result = sorted(rf_result[:6])

    # 회귀 분석 예측
    reg_pred = [model.predict(last_numbers_reshaped)[0] for model in reg_models]
    reg_pred = [max(1, min(45, int(pred))) for pred in reg_pred]
    reg_result = sorted(list(set(reg_pred)))
    while len(reg_result) < 6:
        new_num = np.random.randint(1, 46)
        if new_num not in reg_result:
            reg_result.append(new_num)
    reg_result = sorted(reg_result[:6])
    
    return {
        "LSTM": lstm_result,
        "랜덤 포레스트": rf_result,
        "로지스틱 회귀": reg_result
    }

# 앱 시작시 모델 학습
def initialize_models():
    print("모델 학습을 시작합니다...")
    X, y, last_lotto_numbers = load_and_prepare_data('lotto.xlsx')
    if X is not None:
        lstm_model, rf_models, reg_models = train_models(X, y)
        models['lstm_model'] = lstm_model
        models['rf_models'] = rf_models
        models['reg_models'] = reg_models
        models['last_lotto_numbers'] = last_lotto_numbers
        print("모델 학습이 완료되었습니다!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    # 모델이 없으면 초기화
    if models['lstm_model'] is None:
        try:
            initialize_models()
        except Exception as e:
            return jsonify({'error': f'모델 초기화 중 오류가 발생했습니다: {str(e)}'})
    
    if models['lstm_model'] is None:
        return jsonify({'error': '모델이 아직 준비되지 않았습니다.'})
    
    predictions = predict_next_lotto(
        models['last_lotto_numbers'],
        models['lstm_model'],
        models['rf_models'],
        models['reg_models']
    )
    
    return jsonify({
        'last_numbers': [int(x) for x in models['last_lotto_numbers'].tolist()],
        'predictions': {k: [int(x) for x in v] for k, v in predictions.items()}
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
