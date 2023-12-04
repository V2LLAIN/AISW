import argparse
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def calculate_similarity(file_path1, file_path2, model_path):
    MAX_LEN = 512

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()

    with open(file_path1, 'r', encoding='utf-8') as file1:
        code1 = file1.read()

    with open(file_path2, 'r', encoding='utf-8') as file2:
        code2 = file2.read()
        
    inputs = tokenizer(code1, code2, padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits

    logits = predictions[0]
    probabilities = F.softmax(logits, dim=0)
    similarity_percentage = probabilities[1].item() * 100

    return similarity_percentage

def main():
    parser = argparse.ArgumentParser(description="Code Similarity Evaluation")
    parser.add_argument('--file1', type=str, help='Path to the first Python file')
    parser.add_argument('--file2', type=str, help='Path to the second Python file')
    parser.add_argument('--model_path', type=str, help='Path to the model directory')

    args = parser.parse_args()

    similarity_percentage = calculate_similarity(args.file1, args.file2, args.model_path)
    print(f"Similarity: {similarity_percentage:.2f}%")

if __name__ == "__main__":
    main()


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