import numpy as np
import csv
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy

# -----------------------------
# Fixed parameters
# -----------------------------
EMBEDDING_DIM = 10
DEFAULT_BATCH_SIZE = 16
DEFAULT_LRS = [0.1, 0.01, 0.001]
STABILITY_WINDOW = 5

OUTPUT_DIR = "/Users/rosh_n/Documents/MTECH_NOTES/SEM_2/GAPE/Assignment_Python Coding/Output_Files"

print("\n===== SKIP-GRAM TRAINING =====")

# -----------------------------
# User inputs
# -----------------------------
epochs = int(input("Enter number of epochs (E): "))

batch = input("Enter batch size (press Enter for default = 16): ")
batch_size = int(batch) if batch else DEFAULT_BATCH_SIZE

lr_input = input(
    "Enter learning rates separated by commas "
    "(press Enter for default 0.1,0.01,0.001): "
)

LEARNING_RATES = (
    DEFAULT_LRS if lr_input.strip() == ""
    else [float(lr.strip()) for lr in lr_input.split(",")]
)

# -----------------------------
# Load vocabulary
# -----------------------------
with open(os.path.join(OUTPUT_DIR, "vocab.txt")) as f:
    vocab = [line.split("\t")[0] for line in f]

vocab_size = len(vocab)

# -----------------------------
# Load Skip-gram dataset
# -----------------------------
inputs, outputs = [], []

with open(os.path.join(OUTPUT_DIR, "skipgram_dataset.csv")) as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        inputs.append(int(row[0]))
        outputs.append(int(row[1]))

# -----------------------------
# One-hot encoding
# -----------------------------
X = np.zeros((len(inputs), vocab_size))
Y = np.zeros((len(outputs), vocab_size))

for i in range(len(inputs)):
    X[i, inputs[i]] = 1
    Y[i, outputs[i]] = 1

# -----------------------------
# Train & evaluate each LR
# -----------------------------
results = {}

for lr in LEARNING_RATES:
    print(f"\nTraining Skip-gram with learning rate η = {lr}")

    model = Sequential([
        Dense(EMBEDDING_DIM, input_shape=(vocab_size,), activation="linear"),
        Dense(vocab_size, activation="softmax")
    ])

    model.compile(
        optimizer=SGD(lr),
        loss=CategoricalCrossentropy()
    )

    history = model.fit(
        X, Y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    losses = history.history["loss"]
    final_loss = losses[-1]
    stability = np.var(losses[-STABILITY_WINDOW:])

    embeddings = model.layers[0].get_weights()[0]

    # Save LR-specific outputs
    np.savetxt(
        os.path.join(OUTPUT_DIR, f"embeddings_skipgram_lr_{lr}.csv"),
        embeddings,
        delimiter=","
    )

    with open(os.path.join(OUTPUT_DIR, f"loss_skipgram_lr_{lr}.txt"), "w") as f:
        for l in losses:
            f.write(str(l) + "\n")

    results[lr] = (final_loss, stability, embeddings)

# -----------------------------
# Select best learning rate
# -----------------------------
best_lr = min(results, key=lambda lr: (results[lr][0], results[lr][1]))
best_embeddings = results[best_lr][2]

np.savetxt(
    os.path.join(OUTPUT_DIR, "embeddings_skipgram.csv"),
    best_embeddings,
    delimiter=","
)

print(f"\nBest Skip-gram learning rate selected: η = {best_lr}")
print("Final Skip-gram embeddings saved as embeddings_skipgram.csv")
print("\nSkip-gram training completed for all learning rates.")