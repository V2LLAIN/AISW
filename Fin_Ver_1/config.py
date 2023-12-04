class TrainingConfig:
    learning_rate = 2e-4
    batch_size = 16
    epoch_num = 16
    MAX_LEN = 512
    padding = True
    truncation=True
    return_token_type_ids=True
    model_name = "bert-base-uncased"
    
    """
    BERT: "bert-base-uncased"
    CodeBERT: "microsoft/codebert-base"
    CodeGPT: "microsoft/CodeGPT-small-py"
    CodeBERTa: "huggingface/CodeBERTa-small-v1"
    GraphCodeBERT: "microsoft/graphcodebert-base"
    UnixCoder: "microsoft/unixcoder-base"
    CodeExecutor: "microsoft/codeexecutor"
    LongCoder: "microsoft/longcoder-base"
    """