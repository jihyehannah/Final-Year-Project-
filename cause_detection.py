
# #텍스트 길이
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import pandas as pd
# import torch

# # FLAN-T5 모델과 토크나이저 로드
# model_name = "google/flan-t5-base"  # 모델 이름
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# # GPU 사용 여부 설정
# device = torch.device("cpu")
# model.to(device)

# # 데이터 로드
# data1 = pd.read_csv("test.csv")  # 생성형 데이터
# data2 = pd.read_csv("test2.csv")  # 실생활 데이터

# import matplotlib.pyplot as plt
# import seaborn as sns

# # # 텍스트 길이 계산
# # data1['text_length'] = data1['text'].apply(len)
# # data2['text_length'] = data2['text'].apply(len)

# # # 시각화
# # plt.figure(figsize=(10, 6))
# # sns.kdeplot(data1['text_length'], label='Generated Data', fill=True, alpha=0.5)
# # sns.kdeplot(data2['text_length'], label='Real Data', fill=True, alpha=0.5)
# # plt.title('Text Length Distribution')
# # plt.xlabel('Text Length')
# # plt.ylabel('Density')
# # plt.legend()
# # plt.show()


# # plt.savefig("/data/jigguri/repos/plot.png")
# # print("saved")

# # 단어 빈도

# from sklearn.feature_extraction.text import CountVectorizer

# # CountVectorizer로 단어 빈도 계산
# vectorizer = CountVectorizer(max_features=20)  # 상위 20개 단어
# generated_counts = vectorizer.fit_transform(data1['text'])
# real_counts = vectorizer.fit_transform(data2['text'])

# # 단어 빈도 시각화
# import pandas as pd
# generated_df = pd.DataFrame(generated_counts.toarray(), columns=vectorizer.get_feature_names_out()).sum()
# real_df = pd.DataFrame(real_counts.toarray(), columns=vectorizer.get_feature_names_out()).sum()

# compare_word_counts = pd.DataFrame({'Generated': generated_df, 'Real': real_df})
# compare_word_counts.plot(kind='bar', figsize=(12, 6))
# plt.title('Word Frequency Comparison')
# plt.ylabel('Frequency')
# plt.show()
# plt.savefig("/data/jigguri/repos/word_freq.png")
# print("saved")


# 텍스트 임베딩 비교
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 데이터 로드
data1 = pd.read_csv("test.csv")  # 생성형 데이터
data2 = pd.read_csv("test2.csv")  # 실생활 데이터

# 텍스트 컬럼 확인 및 결측치 제거
data1 = data1.dropna(subset=["text"]).reset_index(drop=True)
data2 = data2.dropna(subset=["text"]).reset_index(drop=True)

# 텍스트 추출
texts_generated = data1['text'].tolist()
texts_real = data2['text'].tolist()

# Sentence-BERT 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')

# 텍스트 임베딩 (numpy 변환)
print("Generating embeddings for Generated Data...")
embeddings_generated = model.encode(texts_generated, convert_to_tensor=True).cpu().numpy()

print("Generating embeddings for Real Data...")
embeddings_real = model.encode(texts_real, convert_to_tensor=True).cpu().numpy()

# 평균 코사인 유사도 계산
cosine_sim_generated = cosine_similarity(embeddings_generated)
cosine_sim_real = cosine_similarity(embeddings_real)

mean_similarity_generated = np.mean(cosine_sim_generated[np.triu_indices_from(cosine_sim_generated, k=1)])
mean_similarity_real = np.mean(cosine_sim_real[np.triu_indices_from(cosine_sim_real, k=1)])

print(f"Mean Cosine Similarity (Generated Data): {mean_similarity_generated:.4f}")
print(f"Mean Cosine Similarity (Real Data): {mean_similarity_real:.4f}")

# t-SNE로 차원 축소
print("Performing t-SNE dimensionality reduction...")
embeddings_combined = np.vstack((embeddings_generated, embeddings_real))
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings_combined)

# 시각화
plt.figure(figsize=(10, 8))
sns.scatterplot(x=embeddings_2d[:len(embeddings_generated), 0], 
                y=embeddings_2d[:len(embeddings_generated), 1], 
                label="Generated Data", alpha=0.7)
sns.scatterplot(x=embeddings_2d[len(embeddings_generated):, 0], 
                y=embeddings_2d[len(embeddings_generated):, 1], 
                label="Real Data", alpha=0.7)
plt.title("t-SNE Visualization of Text Embeddings")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend()
plt.tight_layout()

# 결과 저장
output_dir = "/data/jigguri/repos/"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "embedding_visualization.png")
plt.savefig(output_path)
print(f"Visualization saved to {output_path}")

# 코사인 유사도 결과 저장
similarity_results = {
    "Mean Cosine Similarity (Generated)": mean_similarity_generated,
    "Mean Cosine Similarity (Real)": mean_similarity_real
}
results_df = pd.DataFrame([similarity_results])
results_df.to_csv(os.path.join(output_dir, "cosine_similarity_results.csv"), index=False)
print("Cosine similarity results saved to cosine_similarity_results.csv")
