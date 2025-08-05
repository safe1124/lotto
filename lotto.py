import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# --- 1. 데이터 불러오기 및 전처리 ---

def load_and_prepare_data(file_path='lotto.xlsx'):
    """엑셀 파일에서 로또 데이터를 불러오고 모델 학습에 맞게 전처리합니다."""
    try:
        # 엑셀 파일 읽기
        df = pd.read_excel(file_path)
        print(f"로또 데이터 로드 완료: {len(df)}개 회차 데이터")
        print(f"가장 최근 회차: {df['회차'].max()}회")
        
        # 회차 순서로 정렬 (오래된 것부터)
        df = df.sort_values('회차')
        
        # 당첨 번호만 추출
        numbers = df[['번호1', '번호2', '번호3', '번호4', '번호5', '번호6']].values
    except FileNotFoundError:
        print(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
        return None, None, None
    except Exception as e:
        print(f"오류: 파일을 읽는 중 문제가 발생했습니다. {str(e)}")
        return None, None, None

    # 데이터를 특징(X)과 레이블(y)로 분리
    # 이전 회차(t-1)의 번호를 기반으로 다음 회차(t)의 번호를 예측
    X, y = [], []
    for i in range(len(numbers) - 1):
        X.append(numbers[i])
        y.append(numbers[i+1])
        
    return np.array(X), np.array(y), numbers[-1]

# --- 2. 모델 학습 ---

def train_models(X, y):
    """세 가지 다른 모델을 학습시킵니다."""
    
    # 모델 1: LSTM
    # LSTM은 시퀀스 데이터에 적합하므로 입력 데이터의 형태를 (samples, timesteps, features)로 변경
    X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
    
    lstm_model = Sequential([
        LSTM(128, activation='relu', input_shape=(X_lstm.shape[1], X_lstm.shape[2]), return_sequences=True),
        LSTM(64, activation='relu'),
        Dense(6) # 6개의 번호를 예측
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    print("LSTM 모델 학습 시작...")
    lstm_model.fit(X_lstm, y, epochs=100, batch_size=10, verbose=0)
    print("LSTM 모델 학습 완료.")

    # 모델 2: 랜덤 포레스트 (각 번호 위치별로 별도의 모델 생성)
    print("랜덤 포레스트 모델 학습 시작...")
    rf_models = []
    for i in range(y.shape[1]): # 6개 번호에 대해 각각 학습
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X, y[:, i])
        rf_models.append(model)
    print("랜덤 포레스트 모델 학습 완료.")
    
    # 모델 3: 로지스틱 회귀 (각 번호 위치별로 별도의 모델 생성)
    print("회귀 모델 학습 시작...")
    reg_models = []
    for i in range(y.shape[1]): # 6개 번호에 대해 각각 학습
        model = LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs', multi_class='ovr')
        model.fit(X, y[:, i])
        reg_models.append(model)
    print("회귀 모델 학습 완료.")
    
    return lstm_model, rf_models, reg_models

# --- 3. 다음 회차 번호 예측 ---

def predict_next_lotto(last_numbers, lstm_model, rf_models, reg_models):
    """학습된 모델들로 다음 회차 로또 번호를 예측합니다."""
    
    # 예측을 위해 입력 데이터 형태를 맞춤
    last_numbers_reshaped = last_numbers.reshape(1, 6)
    last_numbers_lstm = last_numbers.reshape(1, 1, 6)
    
    # LSTM 예측
    lstm_pred = lstm_model.predict(last_numbers_lstm, verbose=0)[0]
    # 예측된 실수를 1-45 범위로 제한하고 정수로 변환
    lstm_pred = np.clip(np.round(lstm_pred), 1, 45).astype(int)
    # 중복 제거 후 정렬하고, 6개가 안 되면 랜덤으로 채움
    lstm_result = sorted(list(set(lstm_pred)))
    while len(lstm_result) < 6:
        new_num = np.random.randint(1, 46)
        if new_num not in lstm_result:
            lstm_result.append(new_num)
    lstm_result = sorted(lstm_result[:6])

    # 랜덤 포레스트 예측
    rf_pred = [model.predict(last_numbers_reshaped)[0] for model in rf_models]
    # 1-45 범위로 제한
    rf_pred = [max(1, min(45, int(pred))) for pred in rf_pred]
    rf_result = sorted(list(set(rf_pred)))
    while len(rf_result) < 6:
        new_num = np.random.randint(1, 46)
        if new_num not in rf_result:
            rf_result.append(new_num)
    rf_result = sorted(rf_result[:6])

    # 회귀 분석 예측
    reg_pred = [model.predict(last_numbers_reshaped)[0] for model in reg_models]
    # 1-45 범위로 제한
    reg_pred = [max(1, min(45, int(pred))) for pred in reg_pred]
    reg_result = sorted(list(set(reg_pred)))
    while len(reg_result) < 6:
        new_num = np.random.randint(1, 46)
        if new_num not in reg_result:
            reg_result.append(new_num)
    reg_result = sorted(reg_result[:6])
    
    return {
        "LSTM 예측 번호": lstm_result,
        "랜덤 포레스트 예측 번호": rf_result,
        "회귀 분석 예측 번호": reg_result
    }

# --- 4. 메인 코드 실행 ---

if __name__ == "__main__":
    # 1. 데이터 준비
    X, y, last_lotto_numbers = load_and_prepare_data('lotto.xlsx')
    
    if X is not None:
        # 2. 모델 학습
        lstm_model, rf_models, reg_models = train_models(X, y)
        
        # 3. 예측 실행
        predictions = predict_next_lotto(last_lotto_numbers, lstm_model, rf_models, reg_models)
        
        # 4. 결과 출력
        print("\n" + "="*50)
        print("           로또 번호 예측 결과")
        print("="*50)
        print(f"가장 최근 회차 번호: {last_lotto_numbers.tolist()}")
        print("-"*50)
        for model_name, numbers in predictions.items():
            # numpy 타입을 일반 정수로 변환
            clean_numbers = [int(num) for num in numbers]
            print(f"{model_name}: {clean_numbers}")
        
        print("-"*50)
        print("*주의: 본 예측은 통계적 모델에 기반한 것으로, 당첨을 보장하지 않습니다.")
        print("="*50)

