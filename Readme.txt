EXPERIMENT-6
Implement a text classification model using the Transformer and T5 architecture.
Use the T5 model to classify text into predefined categories (e.g., sentiment classification, spam detection).
a) Fine-tune the T5 model on a labeled dataset.
b) Test the model on unseen text data and classify it.
c) Evaluate the model performance using metrics like accuracy or F1 score.
Program
# =====================================
# 1. INSTALL DEPENDENCIES
# =====================================
!pip -q install transformers datasets scikit-learn
# =====================================
# 2. IMPORT LIBRARIES
# =====================================
import torch
from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# =====================================
# 3. LOAD INBUILT DATASET
# =====================================
dataset = load_dataset("imdb")
# Reduce size for faster training (IMPORTANT for Colab)
dataset = dataset.shuffle(seed=42)
train_data = dataset["train"].select(range(1500))
test_data = dataset["test"].select(range(300))
# =====================================
# 4. LOAD MODEL & TOKENIZER
# =====================================
model_name = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
# =====================================
# 5. PREPROCESS DATA
# =====================================
label_map = {0: "negative", 1: "positive"}
def preprocess(example):
    input_text = "classify sentiment: " + example["text"]
    target_text = label_map[example["label"]]
    inputs = tokenizer(
        input_text,
        padding="max_length",
        truncation=True,
        max_length=128
    )
    targets = tokenizer(
        target_text,
        padding="max_length",
        truncation=True,
        max_length=8
    )
    inputs["labels"] = targets["input_ids"]
    return inputs
train_dataset = train_data.map(preprocess)
test_dataset = test_data.map(preprocess)
train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
# =====================================
# 6. TRAINING (GPU ENABLED)
# =====================================
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    logging_steps=50,
    save_strategy="no",
    fp16=True  # 🔥 Faster training on GPU
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
print("🚀 Training Started...")
trainer.train()
# =====================================
# 7. PREDICTION FUNCTION
# =====================================
def classify(text):
    input_text = "classify sentiment: " + text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    outputs = model.generate(inputs["input_ids"], max_length=10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
# =====================================
# 8. TEST ON UNSEEN DATA
# =====================================
print("\n🔍 Sample Predictions:")
print("Text: I absolutely loved this movie!")
print("Prediction:", classify("I absolutely loved this movie!"))
print("Text: This was the worst film ever.")
print("Prediction:", classify("This was the worst film ever."))
# =====================================
# 9. EVALUATION
# =====================================
true_labels = []
pred_labels = []
print("\n📊 Evaluating...")
for example in test_data.select(range(100)):  # small eval set
    true = label_map[example["label"]]
    pred = classify(example["text"])
    true_labels.append(true)
    pred_labels.append(pred)
accuracy = accuracy_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels, pos_label="positive")
print("\n✅ Results:")
print("Accuracy:", accuracy)
print("F1 Score:", f1)
# =====================================
# 10. SAVE MODEL TO GOOGLE DRIVE (OPTIONAL)
# =====================================
from google.colab import drive
drive.mount('/content/drive')
model.save_pretrained("/content/drive/MyDrive/t5_sentiment_model")
tokenizer.save_pretrained("/content/drive/MyDrive/t5_sentiment_model")
print("\n💾 Model saved to Google Drive!")


Experiment-7
Build a program to extract image features using ResNet50 and generate captions.
Use ResNet50 to extract features from images, and generate captions based on those features.
a) Load a pre-trained ResNet50 model for image feature extraction.
b) Use the features as input to an LSTM or Transformer model for caption generation.
c) Display the generated captions for sample images.
Program
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# ================================================================
#  IMAGE CAPTIONING — ResNet50 + LSTM  [⚡ 5-MIN VERSION]
#  Dataset : Flickr8k  (Kaggle)
#  Encoder : ResNet50  (pretrained, frozen — spatial avg pool 2048)
#  Decoder : LSTM with attention-style merge
#  Target  : Full pipeline under 5 minutes on Kaggle T4 GPU
#
#  SPEED SETTINGS:
#  ┌─────────────────────────────┬──────────┬──────────┐
#  │ Setting                     │ Full     │ 5-min    │
#  ├─────────────────────────────┼──────────┼──────────┤
#  │ Images used                 │ 8,000    │ 1,000    │
#  │ Captions per image          │ 5        │ 2        │
#  │ Vocab size                  │ 10,000   │ 3,000    │
#  │ Max caption length          │ dynamic  │ 25       │
#  │ Feature extraction          │ 1-by-1   │ batch=64 │
#  │ LSTM units                  │ 512      │ 256      │
#  │ Embed dim                   │ 256      │ 128      │
#  │ Training epochs             │ 20       │ 5        │
#  │ Batch size                  │ 32       │ 128      │
#  │ BLEU eval samples           │ 200      │ 50       │
#  └─────────────────────────────┴──────────┴──────────┘
# ================================================================
import os, re, pickle, math, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
nltk.download("punkt", quiet=True)
RUN_START = time.time()
def elapsed(): return f"{(time.time()-RUN_START)/60:.1f} min"
print("TF version :", tf.__version__)
print("GPU        :", tf.config.list_physical_devices("GPU"))

# ── STEP 1: Config ───────────────────────────────────────────────
class CFG:
    BASE_DIR     = "/kaggle/input/datasets/adityajn105/flickr8k"
    IMAGE_DIR    = os.path.join(BASE_DIR, "Images")
    CAPTION_FILE = os.path.join(BASE_DIR, "captions.txt")
    SAVE_DIR     = "/kaggle/working"
    # ⚡ Data limits
    MAX_IMAGES         = 1_000
    CAPTIONS_PER_IMAGE = 2
    # Image
    IMG_SIZE   = (224, 224)
    FEAT_DIM   = 2048          # ResNet50 GlobalAvgPool output
    # Text
    MAX_VOCAB  = 3_000
    MAX_LEN    = 25
    # LSTM model
    EMBED_DIM  = 128           # word embedding size
    LSTM_UNITS = 256           # LSTM hidden units
    DENSE_DIM  = 256           # dense layer before softmax
    DROPOUT    = 0.3
    # Training
    BATCH_SIZE = 128
    EPOCHS     = 5
    LR         = 3e-4
    TRAIN_SPLIT= 0.90
    FEAT_BATCH = 64            # GPU batch for feature extraction
# ── STEP 2: Load & Clean Captions ───────────────────────────────
def load_captions(path, max_images, caps_per_img):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    captions = {}
    for _, row in df.iterrows():
        img = row["image"].strip()
        if len(captions) >= max_images and img not in captions:
            continue
        cap = row["caption"].strip().lower()
        cap = re.sub(r"[^a-z\s]", "", cap)
        cap = "startseq " + cap.strip() + " endseq"
        lst = captions.setdefault(img, [])
        if len(lst) < caps_per_img:
            lst.append(cap)
    return captions
captions = load_captions(CFG.CAPTION_FILE,
                         CFG.MAX_IMAGES, CFG.CAPTIONS_PER_IMAGE)
print(f"[{elapsed()}] ✅ Captions: {len(captions)} images")
# ── STEP 3: Tokenizer ───────────────────────────────────────────
all_captions = [c for caps in captions.values() for c in caps]

tokenizer = Tokenizer(num_words=CFG.MAX_VOCAB, oov_token="<unk>",
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
tokenizer.fit_on_texts(all_captions)
VOCAB_SIZE = min(CFG.MAX_VOCAB, len(tokenizer.word_index)) + 1
MAX_LEN    = CFG.MAX_LEN
print(f"[{elapsed()}] ✅ Vocab: {VOCAB_SIZE}  |  MaxLen: {MAX_LEN}")
# ── STEP 4: ⚡ Batched GPU Feature Extraction (ResNet50 → 2048) ──
def build_resnet_extractor():
    """
    GlobalAveragePooling after ResNet50 conv layers → (2048,) vector.
    Frozen — no fine-tuning needed.
    """
    base = ResNet50(weights="imagenet", include_top=False,
                    input_shape=(*CFG.IMG_SIZE, 3))
    base.trainable = False
    gap  = layers.GlobalAveragePooling2D()(base.output)
    return keras.Model(base.input, gap, name="ResNet50_GAP")
resnet = build_resnet_extractor()
resnet.summary(line_length=65)
def extract_features_batched(image_dir, img_list, batch_size=64):
    """⚡ GPU-batched extraction — much faster than 1-by-1 predict."""
    features = {}
    for start in tqdm(range(0, len(img_list), batch_size),
                      desc="⚡ ResNet50 batch extraction"):
        fnames = img_list[start: start + batch_size]
        arrays, valid = [], []
        for fname in fnames:
            try:
                img = load_img(os.path.join(image_dir, fname),
                               target_size=CFG.IMG_SIZE)
                arr = img_to_array(img)
                arrays.append(arr)
                valid.append(fname)
            except Exception as e:
                print(f"⚠  Skipped {fname}: {e}")
        if not arrays:
            continue
        batch  = preprocess_input(np.array(arrays, dtype=np.float32))
        feats  = resnet.predict(batch, verbose=0)   # (B, 2048)
        for fname, feat in zip(valid, feats):
            features[fname] = feat                   # shape (2048,)
    return features
all_imgs     = list(captions.keys())
img_features = extract_features_batched(CFG.IMAGE_DIR, all_imgs,
                                         CFG.FEAT_BATCH)
print(f"[{elapsed()}] ✅ Features: {len(img_features)} × (2048,)")

# Save for reuse on next run (skip extraction entirely)
with open(os.path.join(CFG.SAVE_DIR, "resnet_lstm_features.pkl"), "wb") as f:
    pickle.dump(img_features, f)
# ── STEP 5: Train / Test Split ──────────────────────────────────
img_names  = list(img_features.keys())
split      = int(CFG.TRAIN_SPLIT * len(img_names))
train_imgs = set(img_names[:split])
test_imgs  = set(img_names[split:])
print(f"Train: {len(train_imgs)}  |  Test: {len(test_imgs)}")
# ── STEP 6: ⚡ Build Training Arrays (teacher forcing) ──────────
def build_sequences(img_subset):
    """
    Teacher-forcing pairs:
      X1 : image feature (2048,)
      X2 : partial caption (MAX_LEN,)  — input to LSTM
      Y  : next word id  (scalar)
    """
    X1, X2, Y = [], [], []
    for img_name in img_subset:
        if img_name not in img_features:
            continue
        feat = img_features[img_name]
        for cap in captions.get(img_name, []):
            seq = tokenizer.texts_to_sequences([cap])[0]
            seq = seq[:MAX_LEN + 1]              # hard-truncate
            for i in range(1, len(seq)):
                in_seq = pad_sequences([seq[:i]], maxlen=MAX_LEN)[0]
                X1.append(feat)
                X2.append(in_seq)
                Y.append(seq[i])
    return (np.array(X1, dtype=np.float32),
            np.array(X2, dtype=np.int32),
            np.array(Y,  dtype=np.int32))
print(f"[{elapsed()}] Building sequences …")
X1_tr, X2_tr, Y_tr = build_sequences(train_imgs)
print(f"[{elapsed()}] ✅ Train samples: {len(Y_tr):,}")
train_ds = (tf.data.Dataset
              .from_tensor_slices(((X1_tr, X2_tr), Y_tr))
              .shuffle(5_000)
              .batch(CFG.BATCH_SIZE)
              .prefetch(tf.data.AUTOTUNE))
# ── STEP 7: ResNet50 + LSTM Caption Model ───────────────────────
#
#   Architecture:
#
#   [Image feature (2048)]          [Partial caption tokens]
#          │                                  │
#     Dense(128, relu)              Embedding(128) + Dropout
#          │                                  │
#          └──────────── Add ─────────────────┘
#                         │
#                    LSTM(256)
#                         │
#                   Dense(256, relu)
#                         │
#                   Dense(VOCAB, softmax)
#
def build_lstm_model():
    # Image branch
    img_input = layers.Input(shape=(CFG.FEAT_DIM,), name="image_feat")
    img_dense = layers.Dense(CFG.EMBED_DIM, activation="relu",
                             name="img_project")(img_input)
    img_drop  = layers.Dropout(CFG.DROPOUT)(img_dense)
    img_rep   = layers.RepeatVector(MAX_LEN)(img_drop)   # (MAX_LEN, EMBED_DIM)

    # Caption branch
    cap_input = layers.Input(shape=(MAX_LEN,), name="cap_input")
    cap_emb   = layers.Embedding(VOCAB_SIZE, CFG.EMBED_DIM,
                                 mask_zero=True,
                                 name="word_embed")(cap_input)
    cap_drop  = layers.Dropout(CFG.DROPOUT)(cap_emb)

    # Merge: element-wise add → LSTM → Dense
    merged    = layers.Add()([img_rep, cap_drop])
    lstm_out  = layers.LSTM(CFG.LSTM_UNITS, name="lstm")(merged)
    lstm_drop = layers.Dropout(CFG.DROPOUT)(lstm_out)
    dense_out = layers.Dense(CFG.DENSE_DIM, activation="relu",
                             name="fc")(lstm_drop)
    output    = layers.Dense(VOCAB_SIZE, activation="softmax",
                             name="output")(dense_out)

    model = keras.Model(inputs=[img_input, cap_input],
                        outputs=output,
                        name="ResNet50_LSTM_Captioner")
    model.compile(
        optimizer=keras.optimizers.Adam(CFG.LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"])
    return model

model = build_lstm_model()
model.summary(line_length=65)
# ── STEP 8: ⚡ Train ─────────────────────────────────────────────
print(f"\n[{elapsed()}] 🚀 Training …")
callbacks = [
    keras.callbacks.EarlyStopping(monitor="loss", patience=2,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5,
                                      patience=1, verbose=1),
    keras.callbacks.ModelCheckpoint(
        os.path.join(CFG.SAVE_DIR, "best_lstm.h5"),
        monitor="loss", save_best_only=True, verbose=0),
]
history = model.fit(train_ds, epochs=CFG.EPOCHS,
                    callbacks=callbacks, verbose=1)
print(f"[{elapsed()}] ✅ Training done")
# ── STEP 9: Greedy Caption Generation ───────────────────────────
def id_to_word(idx):
    return tokenizer.index_word.get(idx, None)

def greedy_caption(model, feat, max_len):
    """Generate caption token-by-token using argmax (greedy)."""
    caption = [tokenizer.word_index.get("startseq", 1)]
    feat_in = feat[None]                             # (1, 2048)
    for _ in range(max_len):
        seq  = pad_sequences([caption], maxlen=max_len)
        pred = model.predict([feat_in, seq], verbose=0)[0]
        nxt  = int(np.argmax(pred))
        word = id_to_word(nxt)
        if word in (None, "endseq"):
            break
        caption.append(nxt)
    return " ".join(id_to_word(i) for i in caption[1:]
                    if id_to_word(i) not in (None, "endseq"))

# ── STEP 10: Beam Search Caption Generation ─────────────────────
def beam_caption(model, feat, max_len, beam_width=3):
    """Beam search for better caption quality."""
    feat_in  = feat[None]
    start_id = tokenizer.word_index.get("startseq", 1)
    end_id   = tokenizer.word_index.get("endseq",   2)
    beams    = [(0.0, [start_id])]
    completed= []
    for _ in range(max_len):
        candidates = []
        for log_p, seq in beams:
            if seq[-1] == end_id:
                completed.append((log_p, seq)); continue
            padded = pad_sequences([seq], maxlen=max_len)
            probs  = model.predict([feat_in, padded], verbose=0)[0]
            log_pr = np.log(probs + 1e-9)
            for idx in np.argsort(log_pr)[-beam_width:]:
                candidates.append((log_p + log_pr[idx],
                                   seq + [int(idx)]))
        if not candidates: break
        beams = sorted(candidates, key=lambda x: x[0],
                       reverse=True)[:beam_width]
    completed += beams
    best = sorted(completed, key=lambda x: x[0], reverse=True)[0][1]
    return " ".join(id_to_word(i) for i in best[1:]
                    if id_to_word(i) not in (None, "endseq"))
# ── STEP 11: ⚡ BLEU Evaluation (50 samples) ─────────────────────
print(f"\n[{elapsed()}] 📊 Evaluating (50 samples) …")
def evaluate(model, captions, features, max_len, n=50):
    actual, predicted = [], []
    for img in list(features.keys())[:n]:
        if img not in captions: continue
        refs = [c.replace("startseq","").replace("endseq","").split()
                for c in captions[img]]
        pred = greedy_caption(model, features[img], max_len).split()
        actual.append(refs); predicted.append(pred)
    sf = SmoothingFunction().method1
    print(f"\n{'─'*35}")
    print(f"  BLEU Scores (greedy, n={n})")
    print(f"{'─'*35}")
    for k, w in enumerate(
            [(1,0,0,0),(.5,.5,0,0),(.33,.33,.33,0),(.25,.25,.25,.25)], 1):
        b = corpus_bleu(actual, predicted, weights=w, smoothing_function=sf)
        bar = "█" * int(b * 40)
        print(f"  BLEU-{k}: {b:.4f}  {bar}")
    print(f"{'─'*35}")
evaluate(model, captions, img_features, MAX_LEN, n=50)
print(f"[{elapsed()}] ✅ Evaluation done")
# ── STEP 12: Visualise Predictions ──────────────────────────────
def visualise(model, captions, features, image_dir, n=4):
    sample = list(features.keys())[:n]
    fig, axes = plt.subplots(2, n, figsize=(n * 5, 8))
    for col, img_name in enumerate(sample):
        img   = Image.open(os.path.join(image_dir, img_name))
        feat  = features[img_name]
        pred_g = greedy_caption(model, feat, MAX_LEN)
        pred_b = beam_caption(model, feat, MAX_LEN, beam_width=3)
        true  = (captions.get(img_name, [""])[0]
                         .replace("startseq","").replace("endseq","").strip())
        # Row 0: image
        axes[0, col].imshow(img)
        axes[0, col].axis("off")
        axes[0, col].set_title(f"Image {col+1}", fontsize=9,
                                fontweight="bold")
        # Row 1: captions text
        axes[1, col].axis("off")
        axes[1, col].text(0.5, 0.75,
            f"🔍 Greedy:\n{pred_g}",
            ha="center", va="center", fontsize=7.5,
            wrap=True, color="#1d3557",
            transform=axes[1, col].transAxes)
        axes[1, col].text(0.5, 0.40,
            f"🎯 Beam:\n{pred_b}",
            ha="center", va="center", fontsize=7.5,
            wrap=True, color="#2a9d8f",
            transform=axes[1, col].transAxes)
        axes[1, col].text(0.5, 0.08,
            f"✅ True:\n{true}",
            ha="center", va="center", fontsize=7.5,
            wrap=True, color="#e63946",
            transform=axes[1, col].transAxes)
    plt.suptitle("ResNet50 + LSTM  |  Image Captioning — Flickr8k",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = os.path.join(CFG.SAVE_DIR, "predictions_lstm.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved {out}")
visualise(model, captions, img_features, CFG.IMAGE_DIR, n=4)
# ── STEP 13: Training Curves ─────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(history.history["loss"], color="#e63946", lw=2.5, marker="o")
ax1.set_title("Training Loss", fontsize=12, fontweight="bold")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.grid(alpha=.3); ax1.set_facecolor("#f8f9fa")
ax2.plot(history.history["accuracy"], color="#2a9d8f", lw=2.5, marker="o")
ax2.set_title("Training Accuracy", fontsize=12, fontweight="bold")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.grid(alpha=.3); ax2.set_facecolor("#f8f9fa")

plt.suptitle("ResNet50 + LSTM — Training Metrics  [⚡ 5-min run]",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(CFG.SAVE_DIR, "training_curves_lstm.png"), dpi=120)
plt.show()
print("✅ Saved training_curves_lstm.png")
# ── STEP 14: Save Model & Tokenizer ─────────────────────────────
model.save(os.path.join(CFG.SAVE_DIR, "resnet50_lstm_model.h5"))
with open(os.path.join(CFG.SAVE_DIR, "tokenizer_lstm.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)
total = (time.time() - RUN_START) / 60
print(f"\n🎉  Total wall-clock time : {total:.1f} minutes")
print("   Outputs in /kaggle/working/:")
print("   ├── resnet50_lstm_model.h5")
print("   ├── best_lstm.h5")
print("   ├── tokenizer_lstm.pkl")
print("   ├── resnet_lstm_features.pkl")
print("   ├── predictions_lstm.png")
print("   └── training_curves_lstm.png")


Experiment-8
Implement a text summarization model using BERT (Bidirectional Encoder Representations from Transformers).
Use BERT to summarize long text documents into shorter versions, preserving the meaning and important information.
a) Load a pre-trained BERT model.
b) Fine-tune the BERT model for summarization tasks.
Program
import pandas as pd
import torch
from transformers import BertTokenizer, EncoderDecoderModel
from transformers import logging
import warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
# 1. Load dataset
data = pd.read_csv("text_summary_dataset.csv").dropna()
#[:2000] → selects only the first 2000 texts from the list
texts = data["text"].tolist()[:2000]
summaries = data["summary"].tolist()[:2000]
# 2. Load tokenizer
#bert-base-uncased is a pretrained BERT model released by Google for natural language processing tasks.
#It reads text from both left and right context simultaneously.
#base" refers to the model size.
#Specifications of bert-base:
#Layers (Transformer blocks): 12
#Hidden size: 768
#Attention heads: 12
#Parameters: ~110 million
#"uncased" means the model does not distinguish uppercase and lowercase letters.
#Hidden size is the dimension (length) of the vector used to represent each token inside the model.
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# 3. Load BERT encoder-decoder model
#A BERT model is used to read the input text (encoder)
#Another BERT model is used to generate the output text (decoder)
# The Hugging Face Transformers library connects them to create a sequence-to-sequence model.
model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "bert-base-uncased",
    "bert-base-uncased"
)
# Required tokens
# Set the token used to start the decoder during generation (uses BERT [CLS] token)
#CLS = Classification token,It is placed at the beginning of a sentence and represents the overall meaning of the sentence.
model.config.decoder_start_token_id = tokenizer.cls_token_id
# Set the token that marks the end of the generated sequence (uses BERT [SEP] token)
#SEP = Separator token,It marks the end of a sentence or separates two sentences.
model.config.eos_token_id = tokenizer.sep_token_id
# Set the padding token so the model ignores padded positions during training
model.config.pad_token_id = tokenizer.pad_token_id
# Ensure the output layer vocabulary size matches the decoder BERT vocabulary
model.config.vocab_size = model.config.decoder.vocab_size
# 4. Tokenize inputs
# Convert the input texts into token IDs that the model can understand
inputs = tokenizer(
    texts,                 # list of original documents
    padding=True,          # add padding tokens so all sequences have the same length
    truncation=True,       # cut the text if it is longer than max_length
    max_length=256,        # maximum number of tokens allowed for input text
    return_tensors="pt"    # return the result as PyTorch tensors
)
# Convert the target summaries into token IDs (these will be used as labels for training)
labels = tokenizer(
    summaries,             # list of target summaries
    padding=True,          # pad shorter summaries to the same length
    truncation=True,       # cut summaries if they exceed max_length
    max_length=64,         # maximum number of tokens allowed for summaries
    return_tensors="pt"    # return as PyTorch tensors
).input_ids                # extract only the token IDs to use as labels
# IMPORTANT FIX (prevents repetition learning)
labels[labels == tokenizer.pad_token_id] = -100
# 5. Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
# 6. Train model
model.train()
# Train the model for 10 epochs (one epoch = one full pass through the dataset)
for epoch in range(10):
    # Forward pass: send input tokens and labels to the model to compute predictions
    outputs = model(
        input_ids=inputs.input_ids,        # token IDs of the input texts
        attention_mask=inputs.attention_mask,  # tells the model which tokens are real and which are padding
        labels=labels                      # target summaries used to compute training loss
    )
    # Extract the loss value (difference between predicted summary and true summary)
    loss = outputs.loss
    # Backward pass: compute gradients for all model parameters
    loss.backward()
    # Update model weights using the optimizer
    optimizer.step()
    # Reset gradients to zero before the next iteration
    optimizer.zero_grad()
    print("Epoch:", epoch+1, "Loss:", loss.item())
# 7. Test summarization
model.eval()
sentence = "Artificial intelligence is transforming healthcare by improving diagnosis and treatment."
test_input = tokenizer(sentence, return_tensors="pt", truncation=True)
# Generate a summary from the input text using the trained model
summary_ids = model.generate(
    test_input.input_ids,              # token IDs of the input sentence
    attention_mask=test_input.attention_mask,  # mask to ignore padding tokens
    max_length=10,                     # maximum number of tokens allowed in the generated summary
    min_length=3,                      # minimum number of tokens required in the summary
    num_beams=4,                       # use beam search with 4 beams to explore multiple possible summaries
    no_repeat_ngram_size=1,            # prevents repeating the same n-gram (sequence of words)
    length_penalty=2.0,                # controls preference for longer or shorter summaries
    early_stopping=True,               # stop generation when all beams reach the end token
    decoder_start_token_id=tokenizer.cls_token_id  # start the decoder with the [CLS] token
)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
clean_summary = summary.replace(" ##", "") # Note the space before ##
print("\nInput Sentence:")
print(sentence)
print("\nGenerated Summary:")
print(summary)
# 7. Test summarization
model.eval()
sentence = "Artificial intelligence is transforming healthcare by improving diagnosis and treatment."
test_input = tokenizer(sentence, return_tensors="pt", truncation=True)
summary_ids = model.generate(
    test_input.input_ids,
    attention_mask=test_input.attention_mask,
    max_length=10,
    min_length=3,
    num_beams=4,
    no_repeat_ngram_size=1,
    length_penalty=2.0,
    early_stopping=True,
    decoder_start_token_id=tokenizer.cls_token_id
)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("\nInput Sentence:")
print(sentence)

print("\nGenerated Summary:")
print(summary)



Experiment-9
Implement a multi-modal deep learning model that combines both image and text features for classification.
Build a multi-modal model that classifies images based on both visual content and associated text (e.g., product images with descriptions).
a) Extract features from images using a pre-trained CNN (e.g., ResNet).
b) Extract text features using an embedding model (e.g., BERT).
c) Combine both features and feed them into a classifier (e.g., a fully connected layer).
d) Train the model and evaluate the classification accuracy.
Program
# =========================
# 🔥 DOWNLOAD DATASET
# =========================
import kagglehub
path = kagglehub.dataset_download("trolukovich/food11-image-dataset")
print("Dataset path:", path)
# =========================
# IMPORTS
# =========================
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
# =========================
# DEVICE
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================
# CONFIG
# =========================
IMG_SIZE = 32
MAX_LEN = 10
VOCAB_SIZE = 1000
NUM_CLASSES = 11
BATCH_SIZE = 16
EPOCHS = 50
# ✅ Class names (IMPORTANT)
class_names = [
    "Bread", "Dairy product", "Dessert", "Egg",
    "Fried food", "Meat", "Noodles/Pasta",
    "Rice", "Seafood", "Soup", "Vegetable/Fruit"
]
train_dir = os.path.join(path, "training")
# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
# =========================
# DATASET
# =========================
class FoodMultiModalDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = transform(image)
        # Simulated text
        text = torch.randint(0, VOCAB_SIZE, (MAX_LEN,))
        return image, text, label
# =========================
# LOAD DATA
# =========================
dataset = FoodMultiModalDataset(train_dir)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
# =========================
# MODEL
# =========================
class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 6 * 6, 64)
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, 64)
    def forward(self, x):
        x = self.embedding(x)
        return x.mean(dim=1)
class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.img = ImageEncoder()
        self.txt = TextEncoder()
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_CLASSES)
        )
    def forward(self, image, text):
        i = self.img(image)
        t = self.txt(text)
        x = torch.cat((i, t), dim=1)
        return self.fc(x)
# =========================
# TRAIN
# =========================
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for images, texts, labels in loader:
        images, texts, labels = images.to(device), texts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
# =========================
# EVALUATE
# =========================
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, texts, labels in loader:
            images, texts, labels = images.to(device), texts.to(device), labels.to(device)
            outputs = model(images, texts)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total
# =========================
# INIT
# =========================
model = MultiModalModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
# =========================
# TRAINING LOOP
# =========================
for epoch in range(EPOCHS):
    loss = train(model, train_loader, optimizer, criterion)
    acc = evaluate(model, val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"   Loss: {loss:.4f}")
    print(f"   Validation Accuracy: {acc:.4f}")
    print("-" * 30)
# =========================
# 🔥 VISUALIZE RESULTS
# =========================
images, texts, labels = next(iter(val_loader))
images = images.to(device)
texts = texts.to(device)
model.eval()
with torch.no_grad():
    outputs = model(images, texts)
    _, preds = torch.max(outputs, 1)
images = images.cpu()
labels = labels.cpu()
preds = preds.cpu()
plt.figure(figsize=(12, 6))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = images[i].permute(1, 2, 0)
    plt.imshow(img)
    true_label = class_names[labels[i]]
    pred_label = class_names[preds[i]]
    plt.title(f"T: {true_label}\nP: {pred_label}")
    plt.axis("off")
plt.tight_layout()
plt.show()
# =========================
# 🔥 FINAL PREDICTION (SINGLE SAMPLE)
# =========================
# Take one sample from validation set
sample_image, sample_text, sample_label = val_dataset[0]
# Add batch dimension
sample_image = sample_image.unsqueeze(0).to(device)
sample_text = sample_text.unsqueeze(0).to(device)
model.eval()
with torch.no_grad():
    output = model(sample_image, sample_text)
    probs = torch.softmax(output, dim=1)
    pred_class = torch.argmax(probs, dim=1).item()
print("\nFinal Prediction:")
print("Actual label:", class_names[sample_label])
print("Predicted class:", pred_class)
print("Predicted label:", class_names[pred_class])
print("Probabilities:", probs.cpu().numpy())



Experiment-10
Build a text generation model using the XLNet architecture.
Fine-tune the XLNet model for text generation, such as creative writing or product descriptions.
a) Load the XLNet pre-trained model.
b) Fine-tune the model with a custom dataset.
Program
!pip install bert-extractive-summarizer==0.7.1
!pip install transformers==2.2.0
!pip install spacy==2.0.12
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from summarizer import Summarizer,TransformerSummarizer
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('water_problem_nlp_en_for_Kaggle_100.csv', delimiter=';', header=0)
df = df.fillna(0)
convert_dict = {'text': str,
                'env_problems': int,
                'pollution': int,
                'treatment': int,
                'climate': int,
                'biomonitoring': int}

df = df.astype(convert_dict)
df = df[:5]
df
df.info()
df['text'].head(10)
df['text'].str.len().max()
# Creation the list with new long block
max_length = 400  # minimum characters in each block
i = 0
bodies = []
while i < len(df):
    body = ""
    body_empty = True
    while (len(body) < max_length) and (i < len(df)):
        if body_empty:
            body = df.loc[i,'text']
            body_empty = False
        else: body += " " + df.loc[i,'text']
        i += 1
    bodies.append(body)
    print("Length of blocks =", len(body))
print(f"\nNumber of text blocks = {len(bodies)}\n")
print("Text blocks:\n", bodies)
# 3. Text Summarizing
min_length_text = 40
bert_summary = []
for i in range(len(bodies)):
    bert_model = Summarizer()
    bert_summary.append(''.join(bert_model(bodies[i], min_length=min_length_text)))
%%time
gpt_summary = []
for i in range(len(bodies)):
    GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
    gpt_summary.append(''.join(GPT2_model(bodies[i], min_length=min_length_text)))
%%time
xlnet_summary = []
for i in range(len(bodies)):
    model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
    xlnet_summary.append(''.join(model(bodies[i], min_length=min_length_text)))

print("All Summarizing Results:\n")
for i in range(len(bodies)):
    print("ORIGINAL TEXT:")
    print(bodies[i])
    print("\nBERT Summarizing Result:")
    print(bert_summary[i])
    print("\nGPT-2 Summarizing Result:")
    print(gpt_summary[i])
    print("\nXLNet Summarizing Result:")
    print(xlnet_summary[i])
    print("\n\n")
