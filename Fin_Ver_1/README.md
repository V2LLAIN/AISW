학습진행하는방법:
python train.py --learning_rate=2e-4 --batch_size=16 --epoch_num=16
(다만 현재로서는 변수에 값입력해도 해당값에따라 값이 변화하지 않아 config.py파일에서 값을 직접 변경할 것.)


추론 및 test진행방법:
    """
    Directory 구조:
    Termproject/
        code_similarity_model/
        code1.py/
        code2.py/
        code3.py/
        code4.py/
        infer.py/    
    """
    
    
    
    """
    python infer.py --file1 ./code1.py --file2 ./code2.py --model_path ./code_similarity_model
    python infer.py --file1 ./code1.py --file2 ./code3.py --model_path ./code_similarity_model
    python infer.py --file1 ./code1.py --file2 ./code4.py --model_path ./code_similarity_model
    python infer.py --file1 ./code2.py --file2 ./code3.py --model_path ./code_similarity_model
    python infer.py --file1 ./code2.py --file2 ./code4.py --model_path ./code_similarity_model
    python infer.py --file1 ./code3.py --file2 ./code4.py --model_path ./code_similarity_model
    """
#  로 실행할 수 있습니다.
