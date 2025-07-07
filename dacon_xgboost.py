import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # 또는 RMSE 등

# CSV 불러오기
df = pd.read_csv("train.csv")

# 결측치 처리 (간단히 제거)
df = df.dropna()

# 타겟과 피처 나누기 (예: 'target'이라는 열이 있다고 가정)
X = df.drop('generated', axis=1)
y = df['generated']

# 학습/검증 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost 모델 학습
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    verbosity=1,
    n_jobs=-1  # 멀티코어 사용
)

model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
print("정확도:", accuracy_score(y_test, y_pred))
