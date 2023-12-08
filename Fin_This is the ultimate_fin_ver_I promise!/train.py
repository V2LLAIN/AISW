import torch
from transformers import Trainer, TrainingArguments
from datasets import load_metric
from model import build_model, get_optimizer_and_scheduler, get_data_collator, compute_metrics
from data import dataset
from config import args


if __name__ == "__main__":
    model, tokenizer = build_model()

    optimizer, scheduler = get_optimizer_and_scheduler(model, args.learning_rate)
    optimizers = optimizer, scheduler
    data_collator = get_data_collator(tokenizer)
    metric = load_metric("accuracy")

    # 학습 인자 설정
    args = TrainingArguments(
        "test",
        save_strategy="epoch",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epoch_num,
        weight_decay=args.weight_decay,
        do_train=True,
        do_eval=True,
        metric_for_best_model="accuracy",
        load_best_model_at_end=True
    )

    encoded_dataset = dataset()
    trainer = Trainer(
        model, args, 
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        optimizers=optimizers
    )

    # 학습 실행
    trainer.train()

"""
학습진행하는방법: 
https://github.com/V2LLAIN/AISW/tree/main/Fin_This%20is%20the%20ultimate_fin_ver_I%20promise!
"""
