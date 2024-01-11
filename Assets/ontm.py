import os
from functools import partial
from transformers import (
    BertForMaskedLM, AdamW, BertConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
)

# from data_collator import *


def tokenize_function(examples, tokenizer):
    text = examples["text"]
    result = tokenizer(text, verbose=False)
    return result


def group_texts(examples, chunk_size:int):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result



    chunk_size:int=512,
    batch_size:int=32,
    lr:float=1e-05,
    weight_decay:float=0.01,
    epochs:int=5,
    num_proc:int=8,

    logging.set_verbosity_debug()

    # オノマトペを登録したトークナイザを使用
    tokenizer = tokenizer
    # tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", trust_remote_code=True)

    dataset = train_dataloader
    # dataset = load_dataset("text", data_files={"train":train_dataset_paths, "validation":valid_dataset_paths}, cache_dir=cache_dir)

    tokenized_datasets = dataset.map(partial(tokenize_function, tokenizer=tokenizer), batched=True, num_proc=num_proc)

    lm_datasets = tokenized_datasets.map(partial(group_texts, chunk_size=chunk_size), batched=True, num_proc=num_proc)

    mask_ids = list(dict_omnt.values())

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mask_ids=mask_ids)

    model = BertForMaskedLM.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

training_args = TrainingArguments(output_dir="output/models/english",
                                 overwrite_output_dir=True,
                                 num_train_epochs=5,
                                 per_gpu_train_batch_size=8,
                                 save_steps = 22222222,
                                 save_total_limit=2)

trainer = Trainer(model=model, args=training_args, data_collator=data_collator, train_dataset=dataset)

trainer.train()
trainer.save_model("output/models/english")