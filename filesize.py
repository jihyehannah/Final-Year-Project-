from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse, sys

parser = argparse.ArgumentParser()
parser.add_argument('-cls', help=' : Please set the class type')
args = parser.parse_args() 
cls_t = args.cls


# 모델 로드
tokenizer = AutoTokenizer.from_pretrained("mohammadtaghizadeh/flan-t5-base-imdb-text-classification")
model = AutoModelForSeq2SeqLM.from_pretrained(f"./results_personality/checkpoint-490")
# model = AutoModelForSeq2SeqLM.from_pretrained(f"./results_multi/checkpoint-2211")
model.to('cuda')

# 모델 크기 계산
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

