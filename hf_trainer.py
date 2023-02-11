from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
)

from datasets.load import load_dataset


# Load tokenizer and model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")


def tokenize(examples, key='text'):
    # from https://huggingface.co/docs/transformers/tasks/language_modeling
    return tokenizer(
        [" ".join(x) for x in examples[key]],
        truncation=True
    )

def group_texts(examples, block_size=128):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

# Load dataset
dataset = load_dataset("imdb")

tok_dataset = dataset.map(
    tokenize,
    batched=True,
    num_proc=4,
    remove_columns=dataset['train'].column_names,
)
lm_dataset = tok_dataset.map(group_texts, batched=True, num_proc=4)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="my_awesome_model",
    evaluation_strategy="steps",
    eval_steps=2000,
    learning_rate=1e-5,
    weight_decay=0.01,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["test"],
    data_collator=data_collator,
)

trainer.train()
