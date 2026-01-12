import re
import csv
from collections import Counter
import os

WINDOW = 4

# FIXED OUTPUT DIRECTORY
OUTPUT_DIR = "/Users/rosh_n/Documents/MTECH_NOTES/SEM_2/GAPE/Assignment_Python Coding/Output_Files"



# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text.split()


# -----------------------------
# Vocabulary
# -----------------------------
def build_vocab(tokens):
    freq = Counter(tokens)
    vocab = {word: idx for idx, word in enumerate(freq)}
    return vocab, freq


def save_vocab(vocab, freq, path):
    with open(path, "w") as f:
        for word, idx in vocab.items():
            f.write(f"{word}\t{idx}\t{freq[word]}\n")


# -----------------------------
# CBOW dataset
# -----------------------------
def build_cbow(tokens, vocab):
    data = []
    for i in range(WINDOW, len(tokens) - WINDOW):
        context = [
            vocab[tokens[j]]
            for j in range(i - WINDOW, i + WINDOW + 1)
            if j != i
        ]
        target = vocab[tokens[i]]
        data.append((context, target))
    return data


# -----------------------------
# Skip-gram dataset
# -----------------------------
def build_skipgram(tokens, vocab):
    data = []
    for i in range(WINDOW, len(tokens) - WINDOW):
        target = vocab[tokens[i]]
        for j in range(i - WINDOW, i + WINDOW + 1):
            if j != i:
                data.append((target, vocab[tokens[j]]))
    return data


# -----------------------------
# CSV writer
# -----------------------------
def save_csv(data, path, mode):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        if mode == "cbow":
            writer.writerow(["context_ids", "target_id"])
            for c, t in data:
                writer.writerow([" ".join(map(str, c)), t])
        else:
            writer.writerow(["input_id", "output_id"])
            for i, o in data:
                writer.writerow([i, o])


# -----------------------------
# Main (Interactive Input)
# -----------------------------
if __name__ == "__main__":

    print("Enter supply chain & logistics text (press ENTER twice to finish):\n")

    lines = []
    while True:
        line = input()
        if line.strip() == "":
            break
        lines.append(line)

    text = " ".join(lines)

    if not text.strip():
        raise ValueError("No input text provided.")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tokens = preprocess(text)
    vocab, freq = build_vocab(tokens)

    save_vocab(vocab, freq, os.path.join(OUTPUT_DIR, "vocab.txt"))

    cbow_data = build_cbow(tokens, vocab)
    skipgram_data = build_skipgram(tokens, vocab)

    save_csv(cbow_data, os.path.join(OUTPUT_DIR, "cbow_dataset.csv"), "cbow")
    save_csv(skipgram_data, os.path.join(OUTPUT_DIR, "skipgram_dataset.csv"), "skipgram")

    print("\nFiles generated successfully")
   
    print(f"Vocabulary size: {len(vocab)}")
    print(f"CBOW samples: {len(cbow_data)}")
    print(f"Skip-gram samples: {len(skipgram_data)}")
