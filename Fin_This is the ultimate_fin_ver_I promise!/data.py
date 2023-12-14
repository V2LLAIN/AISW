import re
#import util 
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from config import args
from model import build_model
from util import main

def preprocess(df,file_name):
  """
  주석의 경우, 코드를 더 잘 설명할 수 있다는 장점은 있으나
  주석이 다른 언어로 되어있는 경우(ex. 한국어, 영어, 일본어 등)를 대비해 주석은 제거.
  """
  preprocess_df = df.replace(re.compile('(#.*)', re.MULTILINE),"",regex=True) #주석 한 줄짜리
  preprocess_df = preprocess_df.replace(re.compile('[\'\"]{3}.*?[\'\"]{3}', re.DOTALL),"",regex=True) #주석 여러줄
  preprocess_df = preprocess_df.replace(re.compile('[\n]{2,}', re.MULTILINE),"\n",regex=True) #다중개행 한번으로
  preprocess_df = preprocess_df.replace(re.compile('[ ]{4}', re.MULTILINE),"\t",regex=True) #tab 변환
  preprocess_df = preprocess_df.replace(re.compile('[ ]{1,3}', re.MULTILINE)," ",regex=True) #공백 여러개 변환
  preprocess_df.to_csv(file_name)
  
def tokenized(examples):
    _ , tokenizer = build_model()
    return tokenizer(examples['code1'],examples['code2'],padding=args.padding,
                     max_length=args.MAX_LEN,truncation=args.truncation,
                     return_token_type_ids=args.return_token_type_ids)

def dataset():
    df = main()
    #df = pd.read_csv('df.csv')
    preprocess(df,"preprocess.csv")
    dataset = load_dataset("csv",data_files="preprocess.csv")['train']
    encoded_dataset = dataset.map(tokenized,remove_columns=['Unnamed: 0','code1','code2'],batched=True)
    encoded_dataset=encoded_dataset.rename_column(original_column_name='similar',new_column_name='labels')
    encoded_dataset = encoded_dataset.train_test_split(0.1,seed=args.seed)
    return encoded_dataset
