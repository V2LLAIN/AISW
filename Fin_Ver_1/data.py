import re
#import util 
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer
from config import TrainingConfig
from model import build_model
from util import main

def preprocess(df,file_name):
  preprocess_df = df.replace(re.compile('(#.*)', re.MULTILINE),"",regex=True) #주석 한 줄
  preprocess_df = preprocess_df.replace(re.compile('[\'\"]{3}.*?[\'\"]{3}', re.DOTALL),"",regex=True) #주석 여러줄
  preprocess_df = preprocess_df.replace(re.compile('[\n]{2,}', re.MULTILINE),"\n",regex=True) #다중개행 한번으로
  preprocess_df = preprocess_df.replace(re.compile('[ ]{4}', re.MULTILINE),"\t",regex=True) #tab 변환
  preprocess_df = preprocess_df.replace(re.compile('[ ]{1,3}', re.MULTILINE)," ",regex=True) #공백 여러개 변환
  preprocess_df.to_csv(file_name)
  
def tokenized(examples):
    _ , tokenizer = build_model()
    return tokenizer(examples['code1'],examples['code2'],padding=TrainingConfig.padding, 
                     max_length=TrainingConfig.MAX_LEN,truncation=TrainingConfig.truncation, 
                     return_token_type_ids=TrainingConfig.return_token_type_ids)

def dataset():
    #df = main()
    # df.csv는 빠른 학습을 위해 미리 preprocessing한 이후 얻은 dataset으로
    # 만약 df.csv가 없다면 위의 df = main()이라는 코드 주석을 해제하고 바로아래줄의 코드를 삭제해주세요.
    df = pd.read_csv('df.csv')
    preprocess(df,"preprocess.csv")
    dataset = load_dataset("csv",data_files="preprocess.csv")['train']
    encoded_dataset = dataset.map(tokenized,remove_columns=['Unnamed: 0','code1','code2'],batched=True)
    encoded_dataset=encoded_dataset.rename_column(original_column_name='similar',new_column_name='labels')
    encoded_dataset = encoded_dataset.train_test_split(0.1,seed=100)
    return encoded_dataset
