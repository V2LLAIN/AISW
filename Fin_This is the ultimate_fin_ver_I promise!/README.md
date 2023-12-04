# 학습진행하는방법:
추가적으로 lr, batch_size, epoch, model, max_len, seed, weight_decay, padding, truncation 또한 terminal에서 학습 시 조정가능.

Additionally, the parameters lr, batch_size, epoch, model, max_len, seed, weight_decay, padding, and truncation can be adjusted during training in the terminal.

### BERT
    python train.py --learning_rate=2e-4 --batch_size=16 --epoch_num=16 --model_name="bert-base-

### CodeBERT
    python train.py --learning_rate=2e-4 --batch_size=16 --epoch_num=16 --model_name="microsoft/codebert-base"

### CodeGPT
    python train.py --learning_rate=2e-4 --batch_size=16 --epoch_num=16 --model_name="microsoft/CodeGPT-small-py"

### CodeBERTa
    python train.py --learning_rate=2e-4 --batch_size=16 --epoch_num=16 --model_name="huggingface/CodeBERTa-small-v1"

### GraphCodeBERT
    python train.py --learning_rate=2e-4 --batch_size=16 --epoch_num=16 --model_name="microsoft/graphcodebert-base"

### UnixCoder
    python train.py --learning_rate=2e-4 --batch_size=16 --epoch_num=16 --model_name="microsoft/unixcoder-base"

### CodeExecutor
    python train.py --learning_rate=2e-4 --batch_size=16 --epoch_num=16 --model_name="microsoft/codeexecutor"

### LongCoder
    python train.py --learning_rate=2e-4 --batch_size=16 --epoch_num=16 --model_name="microsoft/longcoder-base"



  


#
#

<img width="184" alt="스크린샷 2023-12-04 오후 9 14 23" src="https://github.com/V2LLAIN/AISW/assets/104286511/7b346c53-9975-4c04-9708-869fcb711a3e">

# 추론 및 test진행방법:
    
    
    python infer.py --file1 ./code1.py --file2 ./code2.py --model_path ./code_similarity_model
#####    
    python infer.py --file1 ./code1.py --file2 ./code3.py --model_path ./code_similarity_model
#####        
    python infer.py --file1 ./code1.py --file2 ./code4.py --model_path ./code_similarity_model
#####        
    python infer.py --file1 ./code2.py --file2 ./code3.py --model_path ./code_similarity_model
#####        
    python infer.py --file1 ./code2.py --file2 ./code4.py --model_path ./code_similarity_model
#####        
    python infer.py --file1 ./code3.py --file2 ./code4.py --model_path ./code_similarity_model
   
로 실행할 수 있습니다.
