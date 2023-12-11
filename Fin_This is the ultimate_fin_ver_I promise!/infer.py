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
    parser.add_argument('--file1', default="code1.py")
    parser.add_argument('--file2', default="code2.py")
    parser.add_argument('--model_path', default="code_similarity_model")

    args = parser.parse_args()

    similarity_percentage = calculate_similarity(args.file1, args.file2, args.model_path)
    print(f"Similarity: {similarity_percentage:.2f}%")

if __name__ == "__main__":
    main()
"""
추론 및 적용 진행하는방법:
https://github.com/V2LLAIN/AISW/tree/main/Fin_This%20is%20the%20ultimate_fin_ver_I%20promise!
"""
