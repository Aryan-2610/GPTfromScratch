# Understanding `n_embd (C)`, `num_heads`, and `head_size`

Let's break down `n_embd (C)`, `num_heads`, and `head_size` using a clean, non-technical mental model first, and then look at the exact PyTorch math that connects them.

---

# 1. The Real-World Analogy: The Detective Agency

Imagine you are running a private detective agency.

A client walks in and hands you a case file on a suspect.

## `n_embd` (The Case File Size)

This is the total amount of information written in the suspect's dossier.

It contains **32 pages** (or **64, 128, etc.**).

It describes everything about them:

- Height
- Hobbies
- Banking history
- Favorite food

This is your dense token embedding vector (`C`).

---

## `num_heads` (The Number of Detectives)

Instead of forcing one single person to read all **32 pages** and look for every clue simultaneously, you hire a team of **4 specialized detectives**.

---

## `head_size` (The Assignment Size per Detective)

You don't want your detectives tripping over each other reading the same pages.

So, you split the **32-page case file** evenly among them.

- Detective 1 → Pages **1–8** (financial clues)
- Detective 2 → Pages **9–16** (family relationships)
- Detective 3 → Pages **17–24** (travel patterns)
- Detective 4 → Pages **25–32** (criminal records)

Each detective gets exactly **8 pages** to analyze.

This is your `head_size`.

---

# 2. The Golden Equation

Mathematically, these three variables are locked in a strict relationship.

The full embedding dimension is divided equally among all attention heads:

:contentReference[oaicite:0]{index=0}

Example configuration:

```python
n_embd = 32      # Total size of token vector (C)
num_heads = 4    # Number of parallel attention pathways
```

Then:

```python
head_size = 32 // 4
# Each head processes a vector of size 8
```

---

# 3. How the Data Flows Through the Code

Let's visualize how a single token's data moves through your `Block` class using these parameters.

---

## Step A: The Input Vector

A token (for example, the letter `'a'`) enters the block.

It is a single vector of size:

```text
n_embd = 32
```

It contains generalized semantic information about that token.

---

## Step B: The Multi-Head Split

The data hits your `MultiHeadAttention` layer.

The model creates:

```text
4 independent Head objects
```

Each head has its own:

- Query (`Q`) layer
- Key (`K`) layer
- Value (`V`) layer

These layers project the input vector from:

```text
32 → 8
```

because:

```python
head_size = n_embd // num_heads
```

Each head performs self-attention independently in its own restricted **8-dimensional space**.

Example learned behaviors:

- Head 1 → punctuation patterns
- Head 2 → vowel relationships
- Head 3 → sentence structure
- Head 4 → semantic context

---

## Step C: The Grand Concatenation

Once all heads finish computing their outputs:

Each head returns:

```text
Vector size = 8
```

The model concatenates them:

:contentReference[oaicite:1]{index=1}

So you return back to:

```text
n_embd = 32
```

---

# Summary for Notes

### `n_embd`

The total width of the highway.

It is the complete size of the token representation moving between Transformer blocks.

---

### `num_heads`

The number of lanes into which the highway is divided.

Different lanes can learn different patterns in parallel.

---

### `head_size`

The width of one lane.

It is the dimension actually used inside the attention computation:

```text
Q × Kᵀ
```

for a single attention head.

---

## One-Line Intuition

```text
One big representation → split into multiple smaller specialists → process independently → combine back into one big representation
```