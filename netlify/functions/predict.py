import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import warnings
import os

# 경고 메시지 숨기기
warnings.filterwarnings('ignore')

# 전역 변수로 모델 저장
models = {
    'linear_models': None,
    'rf_models': None,
    'reg_models': None,
    'last_lotto_numbers': None
}

def load_and_prepare_data():
    """엑셀 파일에서 로또 데이터를 불러오고 모델 학습에 맞게 전처리합니다."""
    try:
        # Netlify에서는 파일 경로가 다를 수 있음
        file_path = os.path.join(os.path.dirname(__file__), '..', '..', 'lotto.xlsx')
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
    
    # 선형 회귀 모델 (LSTM 대신)
    linear_models = []
    for i in range(y.shape[1]):
        model = LinearRegression()
        model.fit(X, y[:, i])
        linear_models.append(model)

    # 랜덤 포레스트 모델 (경량화)
    rf_models = []
    for i in range(y.shape[1]):
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
        model.fit(X, y[:, i])
        rf_models.append(model)
    
    # 로지스틱 회귀 모델
    reg_models = []
    for i in range(y.shape[1]):
        model = LogisticRegression(max_iter=500, random_state=42, solver='lbfgs', multi_class='ovr')
        model.fit(X, y[:, i])
        reg_models.append(model)
    
    return linear_models, rf_models, reg_models

def predict_next_lotto(last_numbers, linear_models, rf_models, reg_models):
    """학습된 모델들로 다음 회차 로또 번호를 예측합니다."""
    
    last_numbers_reshaped = last_numbers.reshape(1, 6)
    
    # 선형 회귀 예측
    linear_pred = [model.predict(last_numbers_reshaped)[0] for model in linear_models]
    linear_pred = [max(1, min(45, int(round(pred)))) for pred in linear_pred]
    linear_result = sorted(list(set(linear_pred)))
    while len(linear_result) < 6:
        new_num = np.random.randint(1, 46)
        if new_num not in linear_result:
            linear_result.append(new_num)
    linear_result = sorted(linear_result[:6])

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
        "선형 회귀": linear_result,
        "랜덤 포레스트": rf_result,
        "로지스틱 회귀": reg_result
    }

def initialize_models():
    """모델 초기화"""
    global models
    print("모델 학습을 시작합니다...")
    X, y, last_lotto_numbers = load_and_prepare_data()
    if X is not None:
        linear_models, rf_models, reg_models = train_models(X, y)
        models['linear_models'] = linear_models
        models['rf_models'] = rf_models
        models['reg_models'] = reg_models
        models['last_lotto_numbers'] = last_lotto_numbers
        print("모델 학습이 완료되었습니다!")

def handler(event, context):
    """Netlify Functions 핸들러"""
    
    # CORS 헤더 설정
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type',
        'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
    }
    
    # OPTIONS 요청 처리 (CORS preflight)
    if event['httpMethod'] == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': headers,
            'body': ''
        }
    
    try:
        # 모델이 없으면 초기화
        if models['linear_models'] is None:
            initialize_models()
        
        if models['linear_models'] is None:
            return {
                'statusCode': 500,
                'headers': headers,
                'body': json.dumps({'error': '모델이 준비되지 않았습니다.'})
            }
        
        # 예측 실행
        predictions = predict_next_lotto(
            models['last_lotto_numbers'],
            models['linear_models'],
            models['rf_models'],
            models['reg_models']
        )
        
        response_data = {
            'last_numbers': [int(x) for x in models['last_lotto_numbers'].tolist()],
            'predictions': {k: [int(x) for x in v] for k, v in predictions.items()}
        }
        
        return {
            'statusCode': 200,
            'headers': headers,
            'body': json.dumps(response_data)
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': headers,
            'body': json.dumps({'error': f'서버 오류: {str(e)}'})
        }
