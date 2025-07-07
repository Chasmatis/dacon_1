from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
df = pd.read_csv("train.csv")
# 텍스트와 라벨 분리
df['combined'] = df['title'].astype(str) + " [SEP] " + df['full_text'].astype(str)

# TF-IDF로 텍스트를 수치화
vectorizer = TfidfVectorizer(max_features=10000)  # 적절히 조정
X = vectorizer.fit_transform(df['combined'])
y = df['generated']