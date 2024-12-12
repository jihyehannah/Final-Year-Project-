import pandas as pd

# 데이터 불러오기
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 데이터 확장 함수
def expand_data(df):
    expanded_rows = []
    for index, row in df.iterrows():
        expanded_rows.append([row['spontaneity'], "Given the text, identify the spontaneity of the user between planned (0) / na (1) / spontaneous (2).  " + row['text']])
        expanded_rows.append([row['media_sharing'], "Given the text, identify the media_sharing of the user between rarely (0) / na (1) / often (2).  " + row['text']])
        expanded_rows.append([row['spending_habit'], "Given the text, identify the spending_habit of the user between budget (0) / na (1) / fancy (2).  " + row['text']])
        expanded_rows.append([row['scheduling'], "Given the text, identify the scheduling of the user between jammed (0) / na (1) / relaxed (2).  " + row['text']])
        expanded_rows.append([row['personality'], "Given the text, identify the personality of the user between extrovert (0) / na (1) / introvert (2).  " + row['text']])
    
    # 확장된 데이터프레임 생성
    expanded_df = pd.DataFrame(expanded_rows, columns=['multi', 'text'])
    return expanded_df

# 학습 및 테스트 데이터 확장
expanded_train_data = expand_data(train_data)
expanded_test_data = expand_data(test_data)

# 결과 확인
print(expanded_train_data.shape)
print(expanded_test_data.shape)

# 확장된 데이터 저장
expanded_train_data.to_csv('filtered_train_data.csv', index=False)
expanded_test_data.to_csv('filtered_test_data.csv', index=False)



