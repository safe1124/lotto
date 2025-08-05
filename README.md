# 🎱 한국로또분석기

AI 머신러닝을 활용한 한국 로또 번호 예측 웹 애플리케이션

![Python](https://img.shields.io/badge/python-v3.13+-blue.svg)
![Flask](https://img.shields.io/badge/flask-v2.0+-green.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-v2.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📋 프로젝트 소개

한국로또분석기는 실제 로또 당첨 데이터를 분석하여 AI 모델로 다음 회차 번호를 예측하는 웹 애플리케이션입니다. 3가지 다른 머신러닝 알고리즘을 사용하여 다양한 관점에서 번호를 예측합니다.

### ✨ 주요 기능

- 📊 **선형 회귀 모델**: 수학적 패턴 분석으로 번호 예측
- 🌳 **랜덤 포레스트**: 앙상블 학습으로 패턴 인식
- 📈 **로지스틱 회귀**: 통계적 분석으로 확률 계산
- 🎨 **반응형 웹 UI**: 모바일 친화적인 아름다운 디자인
- 📊 **실시간 예측**: 웹에서 즉시 결과 확인
- ☁️ **클라우드 배포**: Netlify를 통한 무료 호스팅

## 🚀 빠른 시작

### 필수 요구사항

- Python 3.13+
- pip 패키지 관리자

### 로컬 실행

1. **저장소 클론**
```bash
git clone https://github.com/safe1124/-.git
cd -
```

2. **가상환경 생성 및 활성화**
```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# 또는 .venv\Scripts\activate  # Windows
```

3. **의존성 설치**
```bash
pip install -r requirements.txt
```

4. **웹 애플리케이션 실행**
```bash
python app.py
```

5. **브라우저에서 접속**
```
http://localhost:5002
```

### Netlify 배포

1. **GitHub 리포지토리 연결**
   - [Netlify](https://netlify.com)에 로그인
   - "New site from Git" 선택
   - GitHub 리포지토리 연결

2. **빌드 설정**
   - Build command: `(공백)`
   - Publish directory: `.`
   - Functions directory: `netlify/functions`

3. **환경 변수 설정**
   ```
   PYTHON_VERSION=3.9
   ```

4. **자동 배포**
   - 설정 완료 후 자동으로 배포됩니다
   - 몇 분 후 고유한 URL로 접속 가능

### 라이브 데모

🌐 **Netlify 배포된 사이트**: [여기에 Netlify URL 추가 예정]

## 📁 프로젝트 구조

```
한국로또분석기/
├── app.py              # Flask 웹 애플리케이션
├── lotto.py            # 로또 예측 모델 (콘솔 버전)
├── lotto.xlsx          # 로또 당첨 데이터 (1183회차)
├── templates/
│   └── index.html      # 웹 UI 템플릿
├── requirements.txt    # Python 의존성
├── .gitignore         # Git 무시 파일
└── README.md          # 프로젝트 문서
```

## 🤖 사용된 AI 모델

### 1. 선형 회귀 (Linear Regression)
- **용도**: 수학적 선형 관계 분석
- **특징**: 과거 번호와 다음 번호 간의 선형 패턴 학습
- **장점**: 빠르고 안정적인 예측

### 2. 랜덤 포레스트 (Random Forest)
- **용도**: 앙상블 학습을 통한 패턴 인식
- **특징**: 각 번호 위치별 독립적 예측
- **파라미터**: 50개 결정 트리, 최대 깊이 10

### 3. 로지스틱 회귀 (Logistic Regression)
- **용도**: 통계적 확률 기반 예측
- **특징**: 다중 클래스 분류 문제로 접근
- **솔버**: LBFGS 최적화

## 📊 데이터셋

- **데이터 소스**: 한국 로또 공식 당첨 번호
- **데이터 범위**: 1회차 ~ 1183회차 (2024년 기준)
- **데이터 형식**: Excel (.xlsx)
- **컬럼**: 회차, 번호1-6, 보너스, 당첨금 정보

## 🎯 예측 방식

1. **데이터 전처리**: 과거 회차 데이터를 시계열로 구성
2. **모델 학습**: 이전 회차 → 다음 회차 패턴 학습
3. **예측 실행**: 최신 회차 데이터로 다음 번호 예측
4. **결과 검증**: 1-45 범위 내 6개 고유 번호 보장

## ⚠️ 주의사항

- 본 프로그램은 **교육 및 연구 목적**으로 제작되었습니다
- 로또는 **순수한 확률 게임**이며, 과거 데이터로는 미래를 정확히 예측할 수 없습니다
- 예측 결과는 **당첨을 보장하지 않습니다**
- 투자는 신중하게 하시기 바랍니다

## 🛠️ 기술 스택

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript
- **AI/ML**: TensorFlow, Scikit-learn
- **Data**: Pandas, NumPy
- **Excel**: openpyxl

## 📈 성능 정보

- **데이터 로딩**: ~1초
- **모델 학습**: ~30초 (1183회차 기준)
- **예측 실행**: ~1초
- **웹 응답**: 실시간

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 👨‍💻 개발자

- **상훈** - 초기 작업 및 전체 개발

## 🙏 감사의 말

- 한국 복권위원회의 공개 당첨 데이터
- TensorFlow 및 Scikit-learn 커뮤니티
- Flask 웹 프레임워크

---

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**
