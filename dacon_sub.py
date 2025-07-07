import pandas as pd

# CSV 파일 읽기
df = pd.read_csv("train.csv")

# 'generated' 열의 값이 1인 행만 필터링
df_generated_1 = df[df['generated'] == 1]

# 결과 출력
print(df_generated_1)

# 만약 결과를 CSV로 저장하고 싶다면:
df_generated_1.to_csv("generated_equals_1.csv", index=False)
