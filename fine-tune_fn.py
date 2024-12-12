from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments 
from transformers import pipeline
# !pip install pytesseract transformers==4.28.1 datasets evaluate rouge-score nltk tensorboard py7zr scikit-learn

from datasets import load_dataset, Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm.auto import tqdm
import numpy as np
from sklearn.metrics import classification_report

import evaluate

import argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument('-cls', help=' : Please set the class type')
args = parser.parse_args() 

#class_type = 'spontaneity' # switch this
print(f"class type: {args.cls}")
class_type = args.cls

classtype2instruction = {
    'personality':"Given the text, identify the personality of the user between extrovert (0) / na (1) / introvert (2). ",
    'spontaneity':"Given the text, identify the spontaneity of the user between planned (0) / na (1) / spontaneous (2). ",
    'media_sharing':"Given the text, identify the media_sharing of the user between rarely (0) / na (1) / often (2). ",
    'spending_habit':"Given the text, identify the spending_habit of the user between budget (0) / na (1) / fancy (2). ",
    'scheduling':"Given the text, identify the scheduling of the user between jammed (0) / na (1) / relaxed (2). "
}

# 데이터 로드
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


dataset = {
    'train': Dataset.from_pandas(train_data),
    'test': Dataset.from_pandas(test_data)
}
dataset['train'] = dataset['train'].shuffle(seed=42)
#dataset['test'] = dataset['test'].shuffle(seed=42)

train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

    #=====dataset adjust======
# 필요한 열 'text'와 'spontaneity'만 남기기


train_df = train_df[['text', class_type]]
test_df = test_df[['text', class_type]]
# 'spontaneity' 열의 이름을 'label'로 변경
#train_df = train_df.rename(columns={'spontaneity': 'label'})
#test_df = test_df.rename(columns={'spontaneity': 'label'})
    #=========================
# #기존 데이터셋 클리어
# dataset.clear()

# Personality 라벨을 문자열로 변환하여 모델 학습에 사용
train_df[class_type] = train_df[class_type].astype(str)
test_df[class_type] = test_df[class_type].astype(str)

# 데이터셋을 Hugging Face Dataset 형식으로 변환
dataset['train'] = Dataset.from_pandas(train_df)
dataset['test'] = Dataset.from_pandas(test_df)

# 모델/토크나이저 설정
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)


# 데이터 전처리
def preprocess_function(examples):
    #inputs = [classtype2instruction[class_type] +" "+ item for item in examples["text"]]
    inputs = [item for item in examples["text"]]

    # 입력 텍스트를 토큰화하고 패딩 및 길이 조정
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=256)
    # 라벨은 이미 0, 1, 2 중 하나로 들어가 있으므로 정수형으로 그대로 사용
    #model_inputs["labels"] = [int(label) for label in examples[class_type]]
    labels = tokenizer(text_target=examples[class_type], padding="max_length", truncation=True, max_length=3)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 전처리 함수 적용
tokenized_dataset = {
    'train': dataset['train'].map(preprocess_function, batched=True, remove_columns=['text', class_type]),
    'test': dataset['test'].map(preprocess_function, batched=True, remove_columns=['text', class_type])
}

# # 텍스트 전처리 함수
# def preprocess_function_text(examples):
#     inputs = ["Given the text, identify the personality of the user between extrovert (0) / na (1) / introvert (2). " + item for item in examples["text"]]
#     model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=221)
#     return model_inputs

# # 레이블 전처리 함수
# def preprocess_function_label(examples):
#     labels = tokenizer(text_target=[str(label) for label in examples[class_type]], padding="max_length", truncation=True, max_length=3)
#     return {"labels": labels["input_ids"]}

# # 전처리 함수 적용
# tokenized_dataset = {
#     'train': dataset['train'].map(lambda x: {**preprocess_function_text(x), **preprocess_function_label(x)}, batched=True, remove_columns=['text', class_type]),
#     'test': dataset['test'].map(lambda x: {**preprocess_function_text(x), **preprocess_function_label(x)}, batched=True, remove_columns=['text', class_type])
# }



#print("===check traindataset===")
#print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

#pipe = pipeline("text2text-generation", model="google/flan-t5-base", device="cuda")

# load model from the hub
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

import evaluate
import nltk
import numpy as np
#from nltk.tokenize import sent_tokenize
#nltk.download("punkt")

# Metric
metric = evaluate.load("f1")

# 수정
def postprocess_text(preds, labels):  # 추가
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    new_preds = []
    for pred in preds:
        if '0' in pred:
            pred = '0'
        elif '1' in pred:
            pred = '1'
        elif '2' in pred:
            pred = '2'
        else:
            pred = '1'
        new_preds.append(pred)
    preds = new_preds

    #preds = [pred.strip() if pred.strip() != '' else '9' for pred in preds]  #빈칸이랑 한글나오는거 수정해야함 
    #labels = [label.strip() if label.strip() != '' else '9' for label in labels]
    
    # rougeLSum expects newline after each sentence
    #preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    #labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

#수정

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    #preds = np.argmax(preds, axis=-1) # token wise 확률 -> token index로 변환
    # if preds.dtype != np.int64:
    #     print("pred wasn't int!!")
    #     print("@@@@@@@@@@pred@@@@@@@@@@")
    #     for i in range(59):
    #         print(f"{preds[i][0]}, {preds[i][1]}, {preds[i][2]}")
    #     #preds = preds.astype(np.int64)
    # if labels.dtype != np.int64:
    #     print("labels wasn't int!!")
    #     print("@@@@@@@@@@labels@@@@@@@@@@")
    #     for i in range(59):
    #         print(f"{labels[i][0]}, {labels[i][1]}, {labels[i][2]}")
    #         labels[i][0], labels[i][1], labels[i][2] = 27, 23, 1
    #     #labels = labels.astype(np.int64)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True) 
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id) 
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    # print("[before postprocessing]")
    #print(f"decoded_preds: {decoded_preds}")
    #print(f"decoded_labels: {decoded_labels}")
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)  #<- interupt here: there are no postproc~~ func
    # print("[after postprocessing]")
    # print(f"decoded_preds: {decoded_preds}")
    # print(f"decoded_labels: {decoded_labels}")
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, average='macro')
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# def compute_metrics(eval_pred):
#     # eval_pred는 logits가 아니라 generated sequences를 포함하므로 다음과 같이 처리
#     predictions, labels = eval_pred
#     # predictions는 생성된 텍스트이므로, 디코딩
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     # labels는 토큰화된 형태이므로, 다시 디코딩
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#     # 정확도 계산 (간단하게 두 텍스트 간의 일치를 비교)
#     accuracy = accuracy_score(decoded_labels, decoded_preds)
#     f1 = f1_score(decoded_labels, decoded_preds, average='weighted')
#     return {"accuracy": accuracy, "f1": f1}

from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

from huggingface_hub import HfFolder
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

# 트레이너 설정
training_args = Seq2SeqTrainingArguments(
    output_dir=f"./results_{class_type}",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    predict_with_generate=True, # 시퀀스 생성 예측
    fp16=False, # Overflows with fp16
    learning_rate=3e-4,
    #weight_decay=0.01,

    num_train_epochs=50,
    logging_dir=f'./logs_{class_type}',
    logging_strategy="epoch",
    # logging_steps=50,

    eval_strategy="epoch",
    save_strategy="epoch",  # 에포크 단위로 체크포인트 저장
    save_total_limit=1,
    load_best_model_at_end= True,
    metric_for_best_model="f1",
)


# Trainer 정의
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics
)

# 모델 학습 시작
trainer.train()


