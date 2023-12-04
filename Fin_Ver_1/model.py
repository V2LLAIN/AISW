import torch
import transformers
import numpy as np
from datasets import load_metric
from config import TrainingConfig
from transformers import DataCollatorWithPadding
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

def build_model():
    model = AutoModelForSequenceClassification.from_pretrained(TrainingConfig.model_name)
    tokenizer = AutoTokenizer.from_pretrained(TrainingConfig.model_name)
    return model, tokenizer

def get_optimizer_and_scheduler(model, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.learning_rate, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1)
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=5)
    return optimizer, scheduler

def get_data_collator(tokenizer):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    return data_collator

def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

