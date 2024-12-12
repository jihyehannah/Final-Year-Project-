from datasets import load_dataset
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import glob
from datasets import load_dataset, Dataset
import datasets
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
from sklearn.metrics import classification_report

import argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument('-cls', help=' : Please set the class type')
args = parser.parse_args() 
cls_t = args.cls

    # 1. Load IMDB Corpus
# 데이터 로드
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# train_data = pd.read_csv('filtered_train_data.csv')
# test_data = pd.read_csv('filtered_test_data.csv')


dataset = {
    'train': Dataset.from_pandas(train_data),
    'test': Dataset.from_pandas(test_data)
}

classtype2instruction = {
    'personality':"Given the text, identify the personality of the user between extrovert (0) / na (1) / introvert (2). ",
    'spontaneity':"Given the text, identify the spontaneity of the user between planned (0) / na (1) / spontaneous (2). ",
    'media_sharing':"Given the text, identify the media_sharing of the user between rarely (0) / na (1) / often (2). ",
    'spending_habit':"Given the text, identify the spending_habit of the user between budget (0) / na (1) / fancy (2). ",
    'scheduling':"Given the text, identify the scheduling of the user between jammed (0) / na (1) / relaxed (2). "
}
#490 140 238 196 294
    # 2. Load fine tune flan t5 model
tokenizer = AutoTokenizer.from_pretrained("mohammadtaghizadeh/flan-t5-base-imdb-text-classification")
#model = AutoModelForSeq2SeqLM.from_pretrained(f"./results_{cls_t}/checkpoint-238")
model = AutoModelForSeq2SeqLM.from_pretrained(f"./results_multi/checkpoint-2211")
model.to('cuda')

    # 3. Test the model
samples_number = len(dataset['test'])
progress_bar = tqdm(range(samples_number))
predictions_list = []
labels_list = []
for i in range(samples_number):
  text = dataset['test']['text'][i]
  inputs = tokenizer.encode_plus(classtype2instruction[cls_t] + text, padding='max_length', max_length=221, return_tensors='pt').to('cuda')
  #inputs = tokenizer.encode_plus(text, padding='max_length', max_length=221, return_tensors='pt').to('cuda')
  outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150, num_beams=4, early_stopping=True)
  prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
#   if prediction == "1":
#      prediction_re = "2"
#   if prediction == "2":
#      prediction_re = "0"
#   if prediction == "0":
#      prediction_re = "1"
  predictions_list.append(prediction)
  labels_list.append(dataset['test'][cls_t][i])

  progress_bar.update(1)

    # 4. Classification report
print("pred - label")
for i in range(59):
    print(f"{int(predictions_list[i])}    {labels_list[i]}")
str_labels_list = []
for i in range(len(labels_list)): str_labels_list.append(str(labels_list[i]))

report = classification_report(str_labels_list, predictions_list)
print(report)

# classification_report를 pandas DataFrame으로 변환
#report = classification_report(str_labels_list, predictions_list, output_dict=True)
#df_report = pd.DataFrame(report).transpose()

# DataFrame을 CSV 파일로 저장
#df_report.to_csv('ver3.classification_media_report.csv', index=True)