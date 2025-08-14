import json
import random

def handler(event, context):
    # 간단한 예시 응답: 실제로는 모델을 불러와 예측을 반환해야 함
    # 여기선 저장된 예측 결과를 사용
    response = {
        "composite": [6,12,19,24,31,33],
        "lstm": [6,12,19,24,31,38],
        "rf": [5,6,12,29,33,44],
        "reg": [1,7,15,26,33,45]
    }
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(response)
    }
