# Deep Learning Lab 5: Word Embeddings with CBOW (Continuous Bag of Words)

## 📚 What This Lab Is About
Welcome to Natural Language Processing (NLP)! This lab teaches how to convert words into numbers that capture their meaning - called **word embeddings**. We use the **CBOW (Continuous Bag of Words)** model to learn these embeddings. This is foundational for all modern NLP applications like chatbots, translation, and sentiment analysis!

---

## 🎯 The Big Picture

### The Challenge: How Do Computers Understand Words?

**The Problem:**
- Computers only understand numbers
- Words are text symbols
- How do we represent "king", "queen", "apple" as numbers?

**Bad Solution: Simple Encoding**
```
king → 1
queen → 2
apple → 3
```
Problems:
- No relationship captured
- "King" isn't closer to "Queen" than to "Apple"
- Math doesn't make sense (king + queen ≠ anything meaningful)

**Good Solution: Word Embeddings**
```
king  → [0.5, 0.8, 0.1, 0.9, ...]  (100 numbers)
queen → [0.5, 0.7, 0.2, 0.8, ...]  (100 numbers)
apple → [0.1, 0.2, 0.9, 0.1, ...]  (100 numbers)
```
Benefits:
- Similar words have similar vectors
- Relationships captured (king - man + woman ≈ queen)
- Can do meaningful math with words!

### What is CBOW?

**CBOW = Continuous Bag of Words**

The idea:
- Look at context words (surrounding words)
- Predict the center word
- Learn word representations in the process

**Example:**
```
Sentence: "The cat sat on the mat"
Context: ["The", "cat", "on", "the"] → Predict: "sat"
Context: ["cat", "sat", "the", "mat"] → Predict: "on"
```

**Analogy:** Like playing a word game where you guess the missing word from the surrounding words. The computer learns what words typically appear together!

---

## 📖 Step-by-Step Explanation

### Step 1: Import Libraries

```python
from keras.preprocessing import text
from keras.preprocessing import sequence
from keras.utils import pad_sequences
from keras.utils import to_categorical
import numpy as np
import pandas as pd
```

**What we're importing:**
- **text**: Tokenization (converting text to numbers)
- **pad_sequences**: Making sequences the same length
- **to_categorical**: Converting labels to one-hot vectors
- **numpy/pandas**: Data manipulation

---

### Step 2: Preparing the Text Data

```python
data = """Deep learning (also known as deep structured learning) is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. 
Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks, convolutional neural networks and Transformers have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, bioinformatics, drug design, medical image analysis, climate science, material inspection and board game programs, where they have produced results comparable to and in some cases surpassing human expert performance.
"""

dl_data = data.split()
```

**What's happening here?**

1. **data**: A paragraph about deep learning
   - Real-world text about neural networks
   - Multiple sentences with technical terms

2. **data.split()**: Split text into words
   - Creates a list of individual words
   - Breaks on whitespace

**Result:** List of words (tokens):
```python
['Deep', 'learning', '(also', 'known', 'as', 'deep', 'structured', ...]
```

---

### Step 3: Tokenization

```python
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(dl_data)
word2id = tokenizer.word_index

word2id['PAD'] = 0
id2word = {v:k for k, v in word2id.items()}
```

**What's happening here?**

#### Creating the Tokenizer
```python
tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(dl_data)
```
- Analyzes all words in the text
- Assigns each unique word a number (ID)
- Most frequent words get lower IDs

#### word2id Dictionary
```python
word2id = tokenizer.word_index
```
This creates a mapping: word → number
```python
{
  'learning': 1,
  'deep': 2,
  'networks': 3,
  'neural': 4,
  'and': 5,
  'as': 6,
  ...
}
```

#### Adding PAD Token
```python
word2id['PAD'] = 0
```
- PAD = Padding token
- Used to make sequences the same length
- Assigned ID 0

#### id2word Dictionary
```python
id2word = {v:k for k, v in word2id.items()}
```
Reverse mapping: number → word
```python
{
  0: 'PAD',
  1: 'learning',
  2: 'deep',
  3: 'networks',
  ...
}
```

#### Converting Words to IDs
```python
wids = [[word2id[w] for w in text.text_to_word_sequence(doc)] for doc in dl_data]
```
Converts each word to its ID:
- "learning" → 1
- "deep" → 2
- "networks" → 3

**Real-world analogy:** 
Like assigning student ID numbers. "John Smith" becomes "12345", making it easier to look up in a database. But we keep a directory (id2word) to convert back!

---

### Step 4: Setting Hyperparameters

```python
vocab_size = len(word2id)      # 75 unique words
embed_size = 100               # Each word → 100 numbers
window_size = 2                # Look 2 words left and right

print('Vocabulary Size:', vocab_size)
print('Vocabulary Sample:', list(word2id.items())[:10])
```

**Key parameters:**

1. **vocab_size = 75**
   - Number of unique words in our text
   - Small vocabulary because we have limited text
   - Real applications: 10,000 - 100,000+ words

2. **embed_size = 100**
   - Each word will be represented by 100 numbers
   - Trade-off: More dimensions = richer meaning but slower training
   - Common choices: 50, 100, 200, 300

3. **window_size = 2**
   - Look at 2 words before and 2 words after target word
   - Total context: 4 words (2 left + 2 right)
   - Larger window = more context but slower training

**Example with window_size=2:**
```
Sentence: "Deep learning is part of machine learning"
                     ↑
              target: "is"
Context: ["Deep", "learning", "of", "machine"]
         [2 left]                   [2 right]
```

---

### Step 5: Generating Training Data

```python
def generate_context_word_pairs(corpus, window_size, vocab_size):
    context_length = window_size * 2
    for words in corpus:
        sentence_length = len(words)
        for index, word in enumerate(words):
            context_words = []
            label_word = []            
            start = index - window_size
            end = index + window_size + 1
            
            context_words.append([words[i] 
                                 for i in range(start, end) 
                                 if 0 <= i < sentence_length 
                                 and i != index])
            label_word.append(word)

            x = pad_sequences(context_words, maxlen=context_length)
            y = to_categorical(label_word, vocab_size)
            yield (x, y)
```

**This is the heart of CBOW! Let's break it down:**

---

#### Understanding the Function

**Purpose:** Create (context, target) pairs for training

**Example walkthrough:**
```
Sentence IDs: [2, 1, 6, 43, 7, ...]
               ↑  ↑  ↑  ↑   ↑
Words:      [deep, learning, is, part, of, ...]
```

**For word "is" (index 2, ID=6):**
```
start = 2 - 2 = 0
end = 2 + 2 + 1 = 5

Context indices: [0, 1, 3, 4]  (skip 2, the target itself)
Context words: [2, 1, 43, 7] → ["deep", "learning", "part", "of"]
Target word: 6 → "is"
```

#### Padding Sequences
```python
x = pad_sequences(context_words, maxlen=context_length)
```

**Why padding?**
- At sentence boundaries, we might have fewer context words
- Need all training samples to be same length
- Pad with 0s (PAD token)

**Example:**
```
Full context: [2, 1, 43, 7]       → Length 4
Edge context: [2, 1]              → Length 2
After padding: [0, 0, 2, 1]       → Length 4 (padded)
```

#### One-Hot Encoding
```python
y = to_categorical(label_word, vocab_size)
```

**Converts target word ID to one-hot vector:**
```
Word ID: 6
Vocab size: 75

One-hot: [0, 0, 0, 0, 0, 0, 1, 0, 0, ..., 0]
                        ↑
                   Position 6
```

**Why one-hot?**
- Neural network needs a format it can learn from
- Can calculate loss (difference between prediction and truth)

---

#### The Generator Pattern
```python
yield (x, y)
```

**What is yield?**
- Creates a generator (memory-efficient)
- Produces one (context, target) pair at a time
- Doesn't load all data into memory at once

**Analogy:** Like a conveyor belt in a factory - items come one at a time rather than all at once!

---

### Step 6: Building the CBOW Model

```python
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Embedding, Lambda

cbow = Sequential()
cbow.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size*2))
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
cbow.add(Dense(vocab_size, activation='softmax'))
cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```

**Let's break down each layer:**

---

#### Layer 1: Embedding Layer

```python
Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size*2)
```

**What is an Embedding Layer?**
- Converts word IDs to dense vectors
- This is where we learn the word embeddings!
- Essentially a lookup table that's learned during training

**Parameters:**
- **input_dim=75**: Vocabulary size (75 unique words)
- **output_dim=100**: Embedding dimension (each word → 100 numbers)
- **input_length=4**: Context size (window_size * 2)

**What it does:**
```
Input: [2, 1, 43, 7]  (4 word IDs)
        ↓
Embedding layer looks up each word:
        ↓
Output: [[0.2, 0.5, ..., 0.8],  ← Embedding for word 2
         [0.3, 0.1, ..., 0.6],  ← Embedding for word 1
         [0.1, 0.9, ..., 0.2],  ← Embedding for word 43
         [0.5, 0.3, ..., 0.7]]  ← Embedding for word 7
         
Shape: (4, 100)  [4 words × 100 dimensions]
```

**Analogy:** Like looking up students in a gradebook. You give student IDs, get back their full grade records.

---

#### Layer 2: Lambda Layer (Averaging)

```python
Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,))
```

**What does this do?**
- Averages all context word embeddings
- Combines 4 vectors (4×100) into 1 vector (100)

**Why average?**
- "Bag of Words" - treat all context words equally
- No regard for order
- Simple but effective!

**Example:**
```
Context embeddings:
Word 1: [0.2, 0.5, 0.8, ...]
Word 2: [0.3, 0.1, 0.6, ...]
Word 3: [0.1, 0.9, 0.2, ...]
Word 4: [0.5, 0.3, 0.7, ...]
         ↓ Average ↓
Result: [0.275, 0.45, 0.575, ...]

Shape: (100,)
```

**Analogy:** Like finding the "average personality" of a friend group to guess who else would fit in.

---

#### Layer 3: Dense Output Layer

```python
Dense(vocab_size, activation='softmax')
```

**What does this do?**
- Takes averaged context embedding (100 numbers)
- Predicts which word should be in the center
- Outputs probability for each word in vocabulary

**Output shape:** (75,) - one probability per word

**Example output:**
```
[0.001, 0.023, 0.001, 0.002, 0.001, 0.650, 0.001, ...]
                                    ↑
                              Word 6: "is"
                        Highest probability!
```

**Softmax:** Converts logits to probabilities (sum to 1)

---

#### Compilation

```python
cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')
```

**loss='categorical_crossentropy':**
- Measures difference between predicted and actual word
- Good for multi-class classification
- Lower loss = better predictions

**optimizer='rmsprop':**
- Learning algorithm
- Automatically adjusts learning rate
- Alternative to Adam, works well for embeddings

---

### Model Summary

```
Layer (type)            Output Shape         Param #
=================================================================
embedding               (None, 4, 100)       7,500
lambda                  (None, 100)          0
dense                   (None, 75)           7,575
=================================================================
Total params: 15,075
```

**Parameters breakdown:**
- **Embedding**: 75 words × 100 dimensions = 7,500
- **Dense**: 100 inputs × 75 outputs + 75 biases = 7,575

**Small model!** Only 15K parameters - trains fast!

---

### Step 7: Training the Model

```python
for epoch in range(1, 6):
    loss = 0.
    i = 0
    for x, y in generate_context_word_pairs(corpus=wids, window_size=window_size, vocab_size=vocab_size):
        i += 1
        loss += cbow.train_on_batch(x, y)
        if i % 100000 == 0:
            print('Processed {} (context, word) pairs'.format(i))

    print('Epoch:', epoch, '\tLoss:', loss)
```

**What's happening here?**

1. **Loop over 5 epochs:**
   - Each epoch = one pass through all data

2. **For each (context, target) pair:**
   - Train model on this pair
   - Accumulate loss

3. **train_on_batch:**
   - Update model weights immediately
   - Don't wait for full batch

**Training progress:**
```
Epoch: 1    Loss: 433.32
Epoch: 2    Loss: 429.14
Epoch: 3    Loss: 425.96
Epoch: 4    Loss: 422.90
Epoch: 5    Loss: 420.44
```

**What we observe:**
- Loss decreases each epoch (learning!)
- Steady improvement
- Model learning word relationships

**What's being learned?**
- The embedding matrix (word vectors)
- Which words appear in similar contexts
- Semantic relationships between words

---

### Step 8: Extracting the Word Embeddings

```python
weights = cbow.get_weights()[0]
weights = weights[1:]  # Skip PAD token
print(weights.shape)   # (74, 100)

pd.DataFrame(weights, index=list(id2word.values())[1:]).head()
```

**What's happening here?**

1. **Get weights from embedding layer:**
   ```python
   weights = cbow.get_weights()[0]
   ```
   - This is the embedding matrix!
   - Each row = embedding for one word

2. **Remove PAD token:**
   ```python
   weights = weights[1:]
   ```
   - PAD (ID=0) isn't a real word
   - Don't need its embedding

3. **Shape:** (74, 100)
   - 74 words (excluding PAD)
   - 100-dimensional embeddings

**The result:** A matrix where each row is a word embedding!
```
            dim_0   dim_1   dim_2  ...  dim_99
learning    0.23    0.15   -0.34  ...   0.12
deep        0.19    0.21   -0.28  ...   0.18
networks    0.31    0.09   -0.41  ...   0.22
...
```

---

## 🔍 Deep Dive: Understanding Word Embeddings

### What Do These Numbers Mean?

**Each dimension captures some aspect of meaning:**
- Dimension 0 might capture "technicality" (0 = common, 1 = technical)
- Dimension 1 might capture "concreteness" (0 = abstract, 1 = concrete)
- Dimension 2 might capture "positive/negative" sentiment
- And so on...

**Example (simplified to 3 dimensions):**
```
king:    [0.8, 0.9, 0.3]  ← High royal, high gender-male, neutral sentiment
queen:   [0.8, 0.1, 0.3]  ← High royal, low gender-male, neutral sentiment
man:     [0.1, 0.9, 0.5]  ← Low royal, high gender-male, positive sentiment
woman:   [0.1, 0.1, 0.5]  ← Low royal, low gender-male, positive sentiment
```

### Famous Word Arithmetic

**king - man + woman ≈ queen**

```
king:   [0.8, 0.9, 0.3]
- man:  [0.1, 0.9, 0.5]
      = [0.7, 0.0, -0.2]
+ woman:[0.1, 0.1, 0.5]
      = [0.8, 0.1, 0.3]  ← Very close to queen!
```

### Measuring Similarity

**Cosine Similarity:**
```
similarity = (A · B) / (||A|| × ||B||)
```

**Similar words have high cosine similarity:**
- "learning" and "training": 0.85
- "deep" and "neural": 0.72
- "learning" and "apple": 0.12

**Analogy:** Like measuring angle between vectors. Small angle = similar meaning!

---

## 🎓 CBOW vs Skip-gram

### CBOW (This Lab)
```
Context: ["the", "cat", "on", "the"]
    ↓
Predict: "sat"
```
- **Given:** Surrounding words
- **Predict:** Center word
- **Good for:** Frequent words
- **Faster:** Trains quicker

### Skip-gram (Alternative)
```
Input: "sat"
    ↓
Predict: ["the", "cat", "on", "the"]
```
- **Given:** Center word
- **Predict:** Surrounding words
- **Good for:** Rare words
- **More accurate:** Usually better embeddings

**Both are part of the Word2Vec family!**

---

## 💡 Real-World Applications

### 1. Search Engines
- Understand query intent
- Find semantically similar documents
- "Best laptop" → also return results for "top computer"

### 2. Recommendation Systems
- Find similar products/movies
- Based on description similarity
- "Action movies with heroes" → Recommend superhero films

### 3. Chatbots & Virtual Assistants
- Understand user queries
- Match questions to answers
- "How's the weather?" ≈ "What's it like outside?"

### 4. Machine Translation
- Initial representation of words
- Transfer meaning across languages
- Foundation for Google Translate

### 5. Sentiment Analysis
- Determine emotion in text
- Product reviews: positive or negative?
- Social media monitoring

### 6. Text Classification
- Spam detection
- Topic categorization
- Content moderation

### 7. Information Retrieval
- Document similarity
- Question answering
- Knowledge bases

---

## 🚀 Advanced Concepts

### Pre-trained Embeddings

**Instead of training from scratch, use pre-trained:**
- **Word2Vec**: Google, 3 billion words
- **GloVe**: Stanford, 840 billion tokens
- **FastText**: Facebook, handles out-of-vocabulary words

**Benefits:**
- Trained on massive datasets
- Better quality embeddings
- Save training time

### Contextual Embeddings

**Limitation of Word2Vec/CBOW:**
```
"Bank" in "river bank" → [0.2, 0.5, ...]
"Bank" in "money bank" → [0.2, 0.5, ...]  ← Same embedding!
```

**Modern solution: BERT, ELMo**
- Different embeddings based on context
- "Bank" has different vectors in different sentences

### Subword Embeddings

**Problem:** What about words not in vocabulary?
```
Training: "running", "runner"
Test: "unrunnable" ← Not seen before!
```

**Solution: FastText**
- Break words into subwords: "un-run-nable"
- Combine subword embeddings
- Can handle any word!

---

## 🔧 Hyperparameters to Tune

### Architecture:
- **embed_size**: 50, 100, 200, 300
  - More = richer representation but slower
  
- **window_size**: 1, 2, 5, 10
  - Larger = more context but less precise

### Training:
- **optimizer**: SGD, Adam, RMSprop
- **learning_rate**: 0.001, 0.01, 0.1
- **epochs**: 5, 10, 20, 100
- **batch_size**: 1 (online), 32, 128, 512

### Data:
- **vocabulary_size**: Limit to most frequent N words
- **min_count**: Ignore words appearing < N times

---

## 🎯 Summary

### What We Learned:

1. ✅ **Word Embeddings:**
   - Convert words to dense vectors
   - Capture semantic meaning
   - Enable word arithmetic

2. ✅ **CBOW Model:**
   - Predict center word from context
   - Learn embeddings during training
   - Simple but effective

3. ✅ **Architecture:**
   - Embedding layer (lookup table)
   - Lambda layer (averaging)
   - Dense layer (prediction)

4. ✅ **Training Process:**
   - Generate (context, target) pairs
   - Update embeddings to minimize loss
   - Similar words end up with similar vectors

5. ✅ **Applications:**
   - Search, translation, chatbots
   - Foundation of modern NLP
   - Transfer to many tasks

---

## 🌟 Key Takeaways

1. **Words need numerical representation** for computers to process them

2. **Embeddings capture meaning** - similar words have similar vectors

3. **Context matters** - words appearing together have similar embeddings

4. **Small vocabulary, big impact** - even with 75 words, we learned the concept

5. **Foundation for NLP** - all modern language AI builds on embeddings

6. **Dimensionality is key** - balance between expressiveness and efficiency

---

## 🎓 Comparison: Different Word Representations

| Method | Size | Example | Similarity |
|--------|------|---------|------------|
| **One-hot** | Vocab size | [0,0,1,0,0] | No similarity |
| **Count vector** | Vocab size | [5,0,12,3,0] | Document-level |
| **TF-IDF** | Vocab size | [0.2,0,0.8,0.1,0] | Better than count |
| **Word2Vec/CBOW** | User defined | [0.2,0.5,...,0.8] | Semantic similarity! |

---

## 🚀 What's Next?

Now that you understand word embeddings:

1. **Try Skip-gram:**
   - Predict context from center word
   - Usually gives better results

2. **Use Pre-trained Embeddings:**
   - Download Word2Vec or GloVe
   - Apply to your own tasks

3. **Explore Modern Methods:**
   - BERT: Contextual embeddings
   - GPT: Generative pre-training
   - Transformers: Attention mechanisms

4. **Build Applications:**
   - Text classifier with embeddings
   - Semantic search engine
   - Recommendation system

5. **Visualize Embeddings:**
   - Use t-SNE or PCA
   - See word clusters in 2D
   - Understand relationships visually

The word embeddings you learned here are the foundation of ChatGPT, Google Translate, and every modern language AI. You've just learned one of the most important concepts in NLP!

Keep exploring the fascinating world of language AI! 🌐✨

