"""
Fake Business Review Detector

Labels: 0 = real, 1 = fake

Requirements:
    pip install transformers datasets scikit-learn torch pandas accelerate
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ─────────────────────────────────────────────
# 1. CONFIGURATION  –
# ─────────────────────────────────────────────
CSV_PATH       = "reviews.csv"
TEXT_COL       = "text"
LABEL_COL      = "label"
MODEL_NAME     = "./fake_review_model"
OUTPUT_DIR     = "./fake_review_model"
MAX_LEN        = 256
BATCH_SIZE     = 16
EPOCHS         = 5
LEARNING_RATE  = 2e-5
TEST_SIZE      = 0.2
SEED           = 42

LABEL_NAMES    = {0: "real", 1: "fake"}

# ─────────────────────────────────────────────
# 2. LOAD & VALIDATE DATASET
# ─────────────────────────────────────────────
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    assert TEXT_COL  in df.columns, f"Column '{TEXT_COL}' not found in CSV."
    assert LABEL_COL in df.columns, f"Column '{LABEL_COL}' not found in CSV."

    df = df[[TEXT_COL, LABEL_COL]].dropna()
    df[LABEL_COL] = df[LABEL_COL].astype(int)

    print(f"\n📂 Loaded {len(df)} reviews")
    print(df[LABEL_COL].value_counts().rename(LABEL_NAMES).to_string())
    return df


# ─────────────────────────────────────────────
# 3. SPLIT & TOKENIZE
# ─────────────────────────────────────────────
def build_dataset(df: pd.DataFrame, tokenizer) -> DatasetDict:
    train_df, eval_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df[LABEL_COL]
    )
    print(f"\n✂️  Train: {len(train_df)} | Eval: {len(eval_df)}")

    def tokenize(batch):
        return tokenizer(
            batch[TEXT_COL],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
        )

    train_ds = Dataset.from_pandas(train_df.rename(columns={LABEL_COL: "labels"}).reset_index(drop=True))
    eval_ds  = Dataset.from_pandas(eval_df.rename(columns={LABEL_COL: "labels"}).reset_index(drop=True))

    tokenized = DatasetDict({"train": train_ds, "eval": eval_ds}).map(tokenize, batched=True)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


# 4. METRICS

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1
    )
    return {
        "accuracy":  round(acc, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }



# 5. TRAIN

def train(tokenized_ds: DatasetDict, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "real", 1: "fake"},
        label2id={"real": 0, "fake": 1},
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",          # evaluate at end of every epoch
        save_strategy="epoch",
        load_best_model_at_end=True,    # keeps the checkpoint with best eval F1
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_steps=10,
        seed=SEED,
        report_to="none",               # set to "wandb" if you want W&B logging
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["eval"],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print(f"\n🚀 Fine-tuning {MODEL_NAME} for {EPOCHS} epochs …\n")
    trainer.train()
    return trainer



# 6. EVALUATE & REPORT RESULTS

def evaluate_and_report(trainer, tokenized_ds: DatasetDict):
    print("\n📊 Final Evaluation on held-out set:\n")
    results = trainer.evaluate()
    for k, v in results.items():
        print(f"  {k:<30} {v}")

    # Detailed confusion matrix + per-class breakdown
    preds_output = trainer.predict(tokenized_ds["eval"])
    preds  = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    print("\n── Confusion Matrix ──────────────────────────")
    cm = confusion_matrix(labels, preds)
    print(f"              Predicted Real  Predicted Fake")
    print(f"  Actual Real       {cm[0][0]:<10}    {cm[0][1]}")
    print(f"  Actual Fake       {cm[1][0]:<10}    {cm[1][1]}")

    print("\n── Per-class Report ──────────────────────────")
    print(classification_report(labels, preds, target_names=["real", "fake"]))

    # Surface the failure cases — mislabeled examples
    eval_df = tokenized_ds["eval"].to_pandas()
    eval_df["predicted"] = preds
    eval_df["actual"]    = labels

    mistakes = eval_df[eval_df["predicted"] != eval_df["actual"]][[TEXT_COL, "actual", "predicted"]]
    mistakes["actual"]    = mistakes["actual"].map(LABEL_NAMES)
    mistakes["predicted"] = mistakes["predicted"].map(LABEL_NAMES)

    print(f"\n── Misclassified Examples ({len(mistakes)} total) ─────")
    for _, row in mistakes.iterrows():
        print(f"  [True: {row['actual']:<4} | Predicted: {row['predicted']:<4}]  {row[TEXT_COL][:90]}")

    return results



# 7. SAVE MODEL + INFERENCE HELPER

def save_model(trainer):
    trainer.save_model(OUTPUT_DIR)
    print(f"\n💾 Model saved to: {OUTPUT_DIR}/")


def predict_review(text: str, model_dir: str = OUTPUT_DIR):
    """
    Quick single-review inference.
    Usage:
        from fake_review_classifier import predict_review
        predict_review("Amazing place, will definitely come back!")
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    with torch.no_grad():
        logits = model(**inputs).logits

    probs     = torch.softmax(logits, dim=-1).squeeze().tolist()
    pred_id   = int(torch.argmax(logits))
    label     = LABEL_NAMES[pred_id]

    print(f"\n🔍 Review  : {text}")
    print(f"   Verdict : {label.upper()}  (real: {probs[0]:.2%} | fake: {probs[1]:.2%})")
    return {"label": label, "real_prob": probs[0], "fake_prob": probs[1]}





if __name__ == "__main__":
    # — Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # — Run pipeline
    df        = load_data(CSV_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds        = build_dataset(df, tokenizer)
    trainer   = train(ds, tokenizer)
    evaluate_and_report(trainer, ds)
    save_model(trainer)

    # — Quick smoke-test predictions
    print("\n── Sample Predictions ────────────────────────")
    predict_review("Amazing food, best restaurant in Charlotte. Highly recommend!")
    predict_review("Absolutely the most incredible dining experience of my entire life. 10 out of 10!!!")
    predict_review("The brisket was a little dry but the service was great and the prices were fair.")


    predict_review("Absolutely phenomenal experience! 5 stars across the board. Highly recommend to everyone!!!")
