import numpy as np
from numpy.linalg import norm
import os

# -----------------------------
# Output directory
# -----------------------------
OUTPUT_DIR = "/Users/rosh_n/Documents/MTECH_NOTES/SEM_2/GAPE/Assignment_Python Coding/Output_Files"

# -----------------------------
# Load vocabulary
# -----------------------------
vocab = []
with open(os.path.join(OUTPUT_DIR, "vocab.txt")) as f:
    for line in f:
        vocab.append(line.split("\t")[0])

vocab_size = len(vocab)

# -----------------------------
# Cosine similarity function
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))


print("\n========== WORD EMBEDDING SIMILARITY EVALUATION ==========")

while True:
    print("\nSelect embedding model:")
    print("1. CBOW")
    print("2. Skip-gram")
    print("0. Exit")

    choice = input("Enter option (1 / 2 / 0): ").strip()

    if choice == "0":
        print("\nExiting similarity evaluation.")
        break

    if choice not in ["1", "2"]:
        print("Invalid option. Please try again.")
        continue

    mode = "cbow" if choice == "1" else "skipgram"
    emb_path = os.path.join(OUTPUT_DIR, f"embeddings_{mode}.csv")

    if not os.path.exists(emb_path):
        print(f"Embeddings for {mode} not found. Train the model first.")
        continue

    embeddings = np.loadtxt(emb_path, delimiter=",")

    # -----------------------------
    # Display vocabulary
    # -----------------------------
    print("\nAvailable vocabulary words:\n")
    for i, word in enumerate(vocab):
        print(f"{word}", end=", ")
        if (i + 1) % 10 == 0:
            print()
    print("\n")

    # -----------------------------
    # Query input
    # -----------------------------
    query_words = input(
        "Enter AT LEAST 5 query words (space-separated): "
    ).lower().split()

    if len(query_words) < 5:
        print("You must enter at least 5 query words. Try again.")
        continue

    # -----------------------------
    # Similarity computation
    # -----------------------------
    print("\n========== COSINE SIMILARITY RESULTS ==========\n")

    with open(os.path.join(OUTPUT_DIR, "similarity_results.txt"), "a") as f:
        f.write(f"\n\nEmbedding Type: {mode.upper()}\n")
        f.write("=" * 50 + "\n")

        for query in query_words:
            if query not in vocab:
                print(f"'{query}' not in vocabulary. Skipped.\n")
                continue

            q_idx = vocab.index(query)
            q_vec = embeddings[q_idx]

            sims = []
            for i, word in enumerate(vocab):
                if word == query:
                    continue
                sim = cosine_similarity(q_vec, embeddings[i])
                sims.append((word, sim))

            top5 = sorted(sims, key=lambda x: -x[1])[:5]

            print(f"Query Word: {query}")
            print("-" * 40)

            f.write(f"\nQuery Word: {query}\n")
            f.write("-" * 40 + "\n")

            for w, s in top5:
                print(f"{w:<15} : {s:.4f}")
                f.write(f"{w:<15} : {s:.4f}\n")

            print()

    print("Results saved. You may continue or exit.")
