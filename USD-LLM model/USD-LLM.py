import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler
import random
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# ================= 配置部分 =================
# 显卡设置：单卡 NVIDIA A6000 (48GB)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 路径设置
DATA_DIR = "dataset"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
VAL_CSV = os.path.join(DATA_DIR, "val.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
# Qwen3-8B 路径
MODEL_PATH = "huggingface/hub/models--Qwen--Qwen3-8B"
OUTPUT_DIR = "output/usd-llm"

RANKING_COLUMN = "Top-6 dimension rankings"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 训练参数 (与 PDF Implementation Details 对齐)
# L_T = L_SFT + λ * L_UP, λ=0.5, Δ=0.2, LoRA rank=16
# batch size=2, lr=1e-4, epochs=3, AdamW
MAX_LENGTH = 2048
BATCH_SIZE = 2
GRAD_ACCUM = 8   # 2 * 8 = 16
LEARNING_RATE = 1e-4
EPOCHS = 3
CONTRASTIVE_ALPHA = 0.5  # λ
CONTRASTIVE_MARGIN = 0.2   # Δ

# ================= 自定义 Trainer =================

class ContrastiveProfileTrainer(Trainer):
    def get_train_dataloader(self):
        """
        重写 DataLoader 获取逻辑，强制关闭 Shuffle，
        并使用 SequentialSampler 保证 Batch 内的偶数/奇数索引是严格配对的。
        注意：必须在数据预处理阶段提前 Shuffle (以 Pair 为单位)。
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator

        train_sampler = SequentialSampler(train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        自定义 Loss：CrossEntropy + Alpha * ContrastiveLoss
        假设：DataCollator 送进来的 Batch，偶数索引 (0, 2...) 是 Profile A，奇数索引 (1, 3...) 是 Profile B。
        """
        outputs = model(**inputs, output_hidden_states=True)
        ce_loss = outputs.loss

        last_hidden_state = outputs.hidden_states[-1]
        batch_size = last_hidden_state.shape[0]

        if batch_size % 2 != 0:
            total_loss = ce_loss
        else:
            seq_lengths = inputs['attention_mask'].sum(dim=1) - 1
            sentence_embeddings = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), seq_lengths]

            embeddings_A = sentence_embeddings[0::2]
            embeddings_B = sentence_embeddings[1::2]

            cosine_sim = F.cosine_similarity(embeddings_A, embeddings_B)
            contrastive_loss = torch.mean(F.relu(cosine_sim - CONTRASTIVE_MARGIN))

            total_loss = ce_loss + CONTRASTIVE_ALPHA * contrastive_loss

        if return_outputs:
            if hasattr(outputs, "hidden_states"):
                outputs.hidden_states = None
            return (total_loss, outputs)

        return total_loss

# ================= 数据预处理 =================

def load_and_pair_data_with_shuffle(csv_paths, shuffle_pairs=True, seed=42):
    """
    读取CSV，构建成对的微调数据集。
    """
    all_pairs = []
    instruction_text = "Based on tweet and user profile, what's the stance of the target, and give the six dimensions of target-specific dimensions that you believe contribute the most to the stance."

    for split, path in csv_paths.items():
        if not os.path.exists(path):
            continue

        print(f"Processing {split} data from {path}...")
        df = pd.read_csv(path)
        df.fillna("", inplace=True)

        grouped = df.groupby(['Target', 'Tweet'])

        for _, group in grouped:
            rows = group.to_dict('records')

            if len(rows) >= 2:
                for i in range(0, len(rows) - 1, 2):
                    row_a = rows[i]
                    row_b = rows[i + 1]

                    sample_a = _create_sample(row_a, instruction_text)
                    sample_b = _create_sample(row_b, instruction_text)

                    all_pairs.append([sample_a, sample_b])

    print(f"Total pairs collected: {len(all_pairs)}.")
    if shuffle_pairs:
        print("Shuffling pairs...")
        random.seed(seed)
        random.shuffle(all_pairs)

    flattened_data = []
    for pair in all_pairs:
        assert pair[0]['raw_tweet'] == pair[1]['raw_tweet'], (
            f"严重错误：Pair中的Tweet不匹配！\n"
            f"A: {str(pair[0]['raw_tweet'])[:50]}...\n"
            f"B: {str(pair[1]['raw_tweet'])[:50]}..."
        )

        del pair[0]['raw_tweet']
        del pair[1]['raw_tweet']

        flattened_data.extend(pair)

    print(f"Final dataset size: {len(flattened_data)} samples. Data alignment check passed.")
    return Dataset.from_list(flattened_data)


def _create_sample(row, instruction):
    return {
        "instruction": instruction,
        "input": f"Target: {row['Target']}\nTweet: {row['Tweet']}\nUser Profile: {row['User_Profile']}",
        "output": f"Stance: {row['Stance']}\nDimensions:\n{row[RANKING_COLUMN]}",
        "raw_tweet": row['Tweet']
    }


def tokenize_dataset(raw_dataset, tokenizer):
    def format_and_tokenize(example):
        messages = [
            {"role": "system", "content": "You are a stance detection expert. Analyze the user's profile and the tweet to determine the stance."},
            {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
            {"role": "assistant", "content": example['output']}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)

        messages_input = [
            {"role": "system", "content": "You are a stance detection expert. Analyze the user's profile and the tweet to determine the stance."},
            {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
        ]
        text_input = tokenizer.apply_chat_template(messages_input, tokenize=False, add_generation_prompt=True)

        tokenized = tokenizer(text, max_length=MAX_LENGTH, truncation=True, padding="max_length")
        tokenized_input = tokenizer(text_input, max_length=MAX_LENGTH, truncation=True, padding=False)

        input_len = len(tokenized_input["input_ids"])

        labels = tokenized["input_ids"].copy()
        labels = [-100 if i < input_len else label for i, label in enumerate(labels)]
        labels = [-100 if mask == 0 else label for i, (label, mask) in enumerate(zip(labels, tokenized["attention_mask"]))]

        tokenized["labels"] = labels
        return tokenized

    print("Tokenizing dataset...")
    return raw_dataset.map(format_and_tokenize, batched=False, remove_columns=raw_dataset.column_names)

# ================= 主流程 =================

def main():
    print(f"Loading Tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    if tokenizer.pad_token is None:
        print("Adding <|pad|> token...")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        need_resize = True
    else:
        need_resize = False

    print("Loading Model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    if need_resize:
        model.resize_token_embeddings(len(tokenizer))

    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    csv_files_train = {"train": TRAIN_CSV}
    csv_files_val = {"val": VAL_CSV}

    raw_train_dataset = load_and_pair_data_with_shuffle(csv_files_train, shuffle_pairs=True)
    raw_val_dataset = load_and_pair_data_with_shuffle(csv_files_val, shuffle_pairs=False)

    train_dataset = tokenize_dataset(raw_train_dataset, tokenizer)
    eval_dataset = tokenize_dataset(raw_val_dataset, tokenizer)

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        optim="adamw_torch",
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        report_to="tensorboard",
        remove_unused_columns=False,
        dataloader_drop_last=True,
    )

    trainer = ContrastiveProfileTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )

    trainer.train_dataset.shuffle = False

    print(f"Train CSV: {TRAIN_CSV}")
    print(f"Val CSV:   {VAL_CSV}")
    print(f"Test CSV:  {TEST_CSV}  (reserved for evaluation)")
    print("Starting Contrastive Training...")
    trainer.train()

    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
