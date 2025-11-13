# Deep Learning Lab 4: Autoencoder for Anomaly Detection

## 📚 What This Lab Is About
This lab introduces **Autoencoders**, a completely different type of neural network that learns to compress and reconstruct data. We use it for **anomaly detection** on ECG (electrocardiogram) data to identify abnormal heartbeats. Unlike previous labs where we classified images, here we're detecting unusual patterns - perfect for finding outliers, fraud, or medical abnormalities!

---

## 🎯 The Big Picture

### What is an Autoencoder?
Imagine you're teaching someone to draw by:
1. Showing them a picture
2. Asking them to memorize the key features
3. Having them redraw it from memory

An autoencoder does exactly this:
- **Encoder**: Compresses input into key features (like memorizing)
- **Decoder**: Reconstructs input from those features (like redrawing)

### Why is this Useful for Anomaly Detection?
- Train the autoencoder on **normal data only**
- It learns what "normal" looks like
- When given abnormal data, it can't reconstruct it well
- Large reconstruction error = anomaly detected!

### The Application: ECG Heart Monitoring
**Goal:** Detect abnormal heartbeats automatically
- **Normal heartbeats**: Autoencoder reconstructs them well
- **Abnormal heartbeats**: Large reconstruction error = flag for review
- **Real-world impact:** Early detection of heart problems!

---

## 📖 Step-by-Step Explanation

### Step 1: Importing Libraries and Loading Data

```python
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import MeanSquaredLogarithmicError

PATH_TO_DATA = 'http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv'
data = pd.read_csv(PATH_TO_DATA, header=None)
```

**What's happening here?**

**New libraries:**
- **pandas**: For handling CSV data (spreadsheet-like format)
- **MinMaxScaler**: To normalize data to [0, 1] range
- **train_test_split**: To split data into training and testing sets
- **MeanSquaredLogarithmicError**: Special loss function for this task

**Loading ECG data:**
- Downloaded from TensorFlow's public datasets
- ECG = electrocardiogram (heart rhythm data)
- Each row is one heartbeat recording

---

### Step 2: Understanding the Dataset

```python
data.shape  # Output: (Number of samples, 141)
```

**Dataset structure:**
- **140 columns**: Features (time-series values representing one heartbeat)
- **1 column** (column 140): Label (0 = abnormal, 1 = normal)

**What does ECG data look like?**
- Time-series recording of electrical heart activity
- Each sample = one heartbeat cycle
- 140 time points per heartbeat
- Values represent electrical voltage

**Real-world analogy:** Like a fitness tracker recording your heart rate, but much more detailed - capturing the exact electrical patterns of each heartbeat.

---

### Step 3: Splitting the Data

```python
features = data.drop(140, axis=1)  # All columns except the last
target = data[140]                 # Just the label column

x_train, x_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, stratify=target
)
```

**What's happening here?**

1. **Separate features from labels:**
   - `features`: The 140 time-series values (the heartbeat pattern)
   - `target`: The label (1 = normal, 0 = abnormal)

2. **train_test_split:**
   - **test_size=0.2**: 80% training, 20% testing
   - **stratify=target**: Keep same proportion of normal/abnormal in both sets

3. **Extract only normal heartbeats for training:**
```python
train_index = y_train[y_train == 1].index  # Indices where label = 1 (normal)
train_data = x_train.loc[train_index]      # Only normal heartbeats
```

**Critical concept:**
We ONLY train on normal heartbeats! The autoencoder learns what "normal" looks like, so it struggles to reconstruct "abnormal" patterns.

**Analogy:** Like a spell-checker trained only on correct English. When it sees a misspelled word, it can't "reconstruct" it well because it never learned incorrect patterns.

---

### Step 4: Scaling the Data

```python
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
x_train_scaled = min_max_scaler.fit_transform(train_data.copy())
x_test_scaled = min_max_scaler.transform(x_test.copy())
```

**What is MinMaxScaler?**
Transforms all features to the range [0, 1]:
```
scaled_value = (value - min) / (max - min)
```

**Example:**
- Original values: [10, 20, 30, 40, 50]
- After scaling: [0.0, 0.25, 0.5, 0.75, 1.0]

**Why scale?**
- Neural networks train better with normalized inputs
- Prevents features with large values from dominating
- Helps gradient descent converge faster

**Important distinction:**
- **fit_transform** on training: Learns min/max from training data
- **transform** on testing: Uses min/max from training (no data leakage!)

---

### Step 5: Creating the Autoencoder Architecture

```python
class AutoEncoder(Model):
  def __init__(self, output_units, ldim=8):
    super().__init__()
    self.encoder = Sequential([
      Dense(64, activation='relu'),
      Dropout(0.1),
      Dense(32, activation='relu'),
      Dropout(0.1),
      Dense(16, activation='relu'),
      Dropout(0.1),
      Dense(ldim, activation='relu')
    ])
    self.decoder = Sequential([
      Dense(16, activation='relu'),
      Dropout(0.1),
      Dense(32, activation='relu'),
      Dropout(0.1),
      Dense(64, activation='relu'),
      Dropout(0.1),
      Dense(output_units, activation='sigmoid')
    ])
  
  def call(self, inputs):
    encoded = self.encoder(inputs)
    decoded = self.decoder(encoded)
    return decoded
```

**This is the heart of our autoencoder! Let's break it down:**

---

#### Understanding the Architecture

**The Encoder (Compression):**
```
Input (140 features)
    ↓
Dense(64) + Dropout → Compress to 64 dimensions
    ↓
Dense(32) + Dropout → Further compress to 32
    ↓
Dense(16) + Dropout → Even more compression to 16
    ↓
Dense(ldim=8) → Bottleneck: Compressed to just 8 numbers!
```

**The Decoder (Reconstruction):**
```
Encoded (8 features)
    ↓
Dense(16) + Dropout → Expand to 16 dimensions
    ↓
Dense(32) + Dropout → Expand to 32
    ↓
Dense(64) + Dropout → Expand to 64
    ↓
Dense(140) → Reconstruct original 140 features
```

**The Complete Flow:**
```
140 numbers → 64 → 32 → 16 → 8 (bottleneck) → 16 → 32 → 64 → 140 numbers
```

---

#### Key Architectural Decisions

**1. Bottleneck (ldim=8):**
- Forces the network to learn the most important 8 features
- Can't just memorize - must learn meaningful patterns!
- Like summarizing a book in one paragraph - you keep only essential info

**2. Symmetric Architecture:**
- Encoder: 64 → 32 → 16 → 8
- Decoder: 8 → 16 → 32 → 64
- Mirror image helps reconstruction

**3. ReLU Activation:**
- Used in hidden layers
- Helps learn non-linear patterns
- Standard choice for hidden layers

**4. Sigmoid Activation (Output):**
- Squashes output to [0, 1] range
- Matches our scaled input data
- Ensures reconstructed values are valid

**5. Dropout(0.1):**
- 10% of neurons randomly turned off during training
- Prevents overfitting
- Makes the model more robust

**Real-world analogy:**
Think of the autoencoder like a game of telephone with a twist:
1. You whisper a long story (140 words) to person 1
2. They summarize it (64 words) and pass it on
3. Each person summarizes more (32, 16, 8 words)
4. At the bottleneck, only 8 words remain!
5. Then each person tries to expand it back
6. Final person reconstructs the original story (140 words)

If the story is familiar (normal heartbeat), they can do this well. If it's weird (abnormal), they struggle!

---

### Step 6: Model Configuration and Training

```python
model = AutoEncoder(output_units=x_train_scaled.shape[1])
model.compile(loss='msle', metrics=['mse'], optimizer='adam')
epochs = 20

history = model.fit(
    x_train_scaled,
    x_train_scaled,  # Note: Same input and output!
    epochs=epochs,
    batch_size=512,
    validation_data=(x_test_scaled, x_test_scaled)
)
```

**What's happening here?**

#### Creating the Model
```python
model = AutoEncoder(output_units=x_train_scaled.shape[1])
```
- `output_units`: 140 (same as input dimension)
- We want to reconstruct the original 140 features

#### Compiling the Model

**loss='msle'** (Mean Squared Logarithmic Error):
- Measures difference between input and reconstruction
- Formula: log(1 + true) - log(1 + predicted)
- **Why logarithmic?** 
  - Less sensitive to large errors
  - Good for values in [0, 1] range
  - Treats relative errors equally (10→20 same as 100→200)

**metrics=['mse']** (Mean Squared Error):
- Another way to measure reconstruction quality
- Formula: (true - predicted)²
- Easier to interpret than MSLE

**optimizer='adam'**:
- Adaptive learning algorithm
- Automatically adjusts learning rate
- Generally works well out of the box

#### Training the Model

**Key observation: Input = Output!**
```python
model.fit(x_train_scaled, x_train_scaled, ...)
```
- We're not predicting labels
- We're trying to reconstruct the input itself
- This is called "unsupervised learning"

**Training parameters:**
- **epochs=20**: See data 20 times
- **batch_size=512**: Process 512 samples at once
- **validation_data**: Test reconstruction on test set

**Training progress:**
```
Epoch 1/20: loss: 0.0112 - mse: 0.0249 - val_loss: 0.0131 - val_mse: 0.0301
Epoch 2/20: loss: 0.0107 - mse: 0.0239 - val_loss: 0.0127 - val_mse: 0.0293
...
Epoch 20/20: loss: 0.0046 - mse: 0.0103 - val_loss: 0.0092 - val_mse: 0.0212
```

**What we observe:**
- Loss decreases: 0.0112 → 0.0046 (getting better!)
- Validation loss also decreases: 0.0131 → 0.0092
- Model learns to reconstruct normal heartbeats accurately

---

### Step 7: Visualizing Training Progress

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel('MSLE Loss')
plt.legend(['loss', 'val_loss'])
plt.show()
```

**What to look for:**
- Both lines going down = good learning
- Lines close together = not overfitting
- Flattening at end = model converged

**Interpretation:**
- Training loss (blue): How well it reconstructs training data
- Validation loss (orange): How well it reconstructs test data
- Both decrease smoothly = healthy training!

---

### Step 8: Finding the Anomaly Threshold

```python
def find_threshold(model, x_train_scaled):
  reconstructions = model.predict(x_train_scaled)
  reconstruction_errors = tf.keras.losses.msle(reconstructions, x_train_scaled)
  threshold = np.mean(reconstruction_errors.numpy()) \
   + np.std(reconstruction_errors.numpy())
  return threshold

threshold = find_threshold(model, x_train_scaled)
print(f"Threshold: {threshold}")  # Output: 0.009583...
```

**What's happening here?**

1. **Get reconstructions:**
   - Pass normal training data through autoencoder
   - Get reconstructed versions

2. **Calculate reconstruction errors:**
   - Compare original vs reconstructed
   - Larger error = worse reconstruction

3. **Set threshold:**
   ```
   threshold = mean(errors) + std(errors)
   ```
   - **Mean**: Average reconstruction error for normal data
   - **Std**: Standard deviation (how spread out errors are)
   - **Threshold**: One standard deviation above mean
   - Captures ~84% of normal data

**The logic:**
- Normal heartbeats: Error < threshold (reconstructed well)
- Abnormal heartbeats: Error > threshold (reconstructed poorly)

**Analogy:** 
Imagine grading essays where students learned to write one style:
- Essays in learned style: Easy to "reconstruct" (paraphrase)
- Essays in different style: Hard to reconstruct accurately
- Set threshold based on: "How different is too different?"

---

### Step 9: Making Predictions

```python
def get_predictions(model, x_test_scaled, threshold):
  predictions = model.predict(x_test_scaled)
  errors = tf.keras.losses.msle(predictions, x_test_scaled)
  anomaly_mask = pd.Series(errors) > threshold
  preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
  return preds

predictions = get_predictions(model, x_test_scaled, threshold)
```

**What's happening here?**

1. **Get reconstructions for test data:**
   ```python
   predictions = model.predict(x_test_scaled)
   ```

2. **Calculate reconstruction errors:**
   ```python
   errors = tf.keras.losses.msle(predictions, x_test_scaled)
   ```

3. **Create anomaly mask:**
   ```python
   anomaly_mask = pd.Series(errors) > threshold
   ```
   - True = error exceeds threshold (anomaly)
   - False = error below threshold (normal)

4. **Convert to labels:**
   ```python
   preds = anomaly_mask.map(lambda x: 0.0 if x == True else 1.0)
   ```
   - Anomaly (True) → 0
   - Normal (False) → 1
   - Matches original labels!

**The decision rule:**
```
If reconstruction_error > threshold:
    Label = 0 (Abnormal heartbeat)
Else:
    Label = 1 (Normal heartbeat)
```

---

### Step 10: Evaluating Performance

```python
accuracy_score(predictions, y_test)
```

**What does this tell us?**
- Compares our predictions against true labels
- Returns percentage of correct classifications
- Higher = better anomaly detection

**Typical results:** 85-95% accuracy
- Most normal heartbeats correctly identified as normal
- Most abnormal heartbeats correctly identified as abnormal

**Confusion Matrix (conceptual):**
```
                Predicted Normal  Predicted Abnormal
Actually Normal      True Neg         False Pos
Actually Abnormal    False Neg        True Pos
```

**Why not 100% accuracy?**
- Some abnormal patterns might be close to normal
- Some normal patterns might be unusual
- Real medical data is noisy and complex
- This is actually excellent performance for medical data!

---

## 🔍 Deep Dive: How Autoencoders Detect Anomalies

### The Reconstruction Error Concept

**For Normal Data:**
```
Input: [0.2, 0.5, 0.8, 0.3, ...]  (Normal heartbeat)
    ↓
Encoder: [0.15, 0.72, 0.33, 0.91, 0.28, 0.64, 0.19, 0.55]  (8 features)
    ↓
Decoder: [0.19, 0.51, 0.79, 0.31, ...]  (Reconstructed)
    ↓
Error: Very small! (0.003)
```
Low error → Classified as normal ✓

**For Abnormal Data:**
```
Input: [0.1, 0.9, 0.2, 0.7, ...]  (Abnormal heartbeat)
    ↓
Encoder: [0.45, 0.22, 0.88, 0.11, 0.93, 0.34, 0.67, 0.20]
    ↓
Decoder: [0.3, 0.6, 0.4, 0.5, ...]  (Poor reconstruction)
    ↓
Error: Large! (0.025)
```
High error → Classified as abnormal ✓

### Why The Bottleneck Matters

**Without bottleneck (140 → 140):**
- Network could just memorize (identity function)
- Would reconstruct everything perfectly
- No anomaly detection!

**With bottleneck (140 → 8 → 140):**
- Must compress to 8 most important features
- Can only remember key patterns
- Novel patterns (anomalies) can't be compressed/reconstructed well

**Analogy:** 
Imagine you can only remember 8 characteristics of normal faces:
- You can reconstruct similar faces well
- But an alien face? You'd struggle because it doesn't fit your learned patterns!

---

## 🎓 Key Concepts Explained

### Unsupervised Learning
- **Supervised**: Learn from labeled examples (input → label)
- **Unsupervised**: Learn patterns without labels (input → patterns)
- Autoencoders are unsupervised (self-supervised)

### Dimensionality Reduction
- Compress 140 features → 8 features
- Keep most important information
- Similar to PCA, but non-linear and more powerful

### Anomaly Detection Approaches

**1. Statistical:**
- Assume normal data follows a distribution
- Flag outliers

**2. Distance-based:**
- Normal data clusters together
- Anomalies are far from cluster

**3. Reconstruction-based (Autoencoders):**
- Learn to reconstruct normal data
- Anomalies have high reconstruction error

### One-Class Classification
- Only train on one class (normal)
- Everything else is "not normal"
- Perfect when abnormal examples are rare

---

## 💡 Real-World Applications

### Medical Diagnosis
1. **ECG Monitoring** (this lab):
   - Detect arrhythmias
   - Alert doctors to unusual patterns
   - Early heart problem detection

2. **Brain Activity (EEG):**
   - Detect seizures
   - Monitor sleep disorders

3. **Medical Imaging:**
   - Detect tumors in MRI/CT scans
   - Find abnormalities in X-rays

### Industrial Applications

1. **Manufacturing:**
   - Detect defective products
   - Quality control on assembly lines

2. **Predictive Maintenance:**
   - Identify failing machinery
   - Prevent costly breakdowns

3. **Network Security:**
   - Detect intrusions
   - Identify unusual network traffic

### Financial Services

1. **Fraud Detection:**
   - Unusual credit card transactions
   - Suspicious bank account activity

2. **Market Monitoring:**
   - Detect market manipulation
   - Identify unusual trading patterns

### Other Applications

- **Video Surveillance**: Detect unusual behavior
- **IoT Sensors**: Identify sensor failures
- **Cybersecurity**: Detect malware and attacks

---

## 🚀 Advantages and Limitations

### Advantages of Autoencoders

✅ **No labeled anomaly data needed**
- Only need examples of normal data
- Perfect when anomalies are rare

✅ **Learns complex patterns**
- Can capture non-linear relationships
- More powerful than simple statistics

✅ **Dimensionality reduction**
- Compresses high-dimensional data
- Extracts meaningful features

✅ **Scalable**
- Can handle large datasets
- Parallel processing on GPUs

### Limitations

❌ **Requires lots of normal data**
- Needs sufficient examples to learn patterns

❌ **Threshold selection is tricky**
- Must balance false positives vs false negatives
- May need domain expertise

❌ **Can miss subtle anomalies**
- If anomaly is similar to normal data

❌ **Black box nature**
- Hard to explain WHY something is anomalous

---

## 🔧 Hyperparameters to Tune

### Architecture:
- **Bottleneck size (ldim)**: 4, 8, 16, 32
  - Smaller = more compression = better anomaly detection
  - Larger = less compression = better reconstruction
  
- **Layer sizes**: [64, 32, 16] vs [128, 64, 32]
  - More neurons = more capacity = better complex patterns

- **Dropout rate**: 0.0, 0.1, 0.2, 0.5
  - Higher = more regularization = less overfitting

### Training:
- **Learning rate**: Often 0.001 (default for Adam)
- **Batch size**: 128, 256, 512
- **Epochs**: 10, 20, 50

### Threshold:
- **mean + std** (current approach)
- **mean + 2*std** (stricter)
- **mean + 0.5*std** (more sensitive)
- **Percentile-based**: 95th percentile of errors

---

## 🎯 Summary

### What We Learned:

1. ✅ **Autoencoders for Anomaly Detection:**
   - Train only on normal data
   - Learn to compress and reconstruct
   - High reconstruction error = anomaly

2. ✅ **Architecture:**
   - Encoder: Compress 140 → 8 features
   - Decoder: Reconstruct 8 → 140 features
   - Bottleneck forces learning of key patterns

3. ✅ **Training Process:**
   - Input = Output (unsupervised)
   - MSLE loss measures reconstruction quality
   - Model learns what "normal" looks like

4. ✅ **Anomaly Detection:**
   - Set threshold based on reconstruction errors
   - Error > threshold = anomaly
   - Achieved good accuracy on ECG data

5. ✅ **Real-World Impact:**
   - Early medical diagnosis
   - Fraud detection
   - Quality control
   - Security monitoring

---

## 🌟 Key Takeaways

1. **Unsupervised learning is powerful** when you don't have labeled anomaly examples

2. **The bottleneck is crucial** - it forces the network to learn compressed representations

3. **Reconstruction error is the key** - normal data reconstructs well, anomalies don't

4. **Threshold selection matters** - balance between false positives and false negatives

5. **Real-world applications are vast** - any domain with "normal" patterns and rare anomalies

---

## 🎓 Comparison with Other Methods

| Method | Pros | Cons |
|--------|------|------|
| **Autoencoders** | Learns complex patterns, no labeled anomalies needed | Requires tuning, black box |
| **Statistical (Z-score)** | Simple, interpretable | Assumes distribution, only linear |
| **Isolation Forest** | Fast, handles high dimensions | May miss local anomalies |
| **One-Class SVM** | Solid mathematical foundation | Slow on large datasets |
| **LSTM Autoencoders** | Great for time-series | More complex, harder to train |

---

## 🚀 What's Next?

Now that you understand autoencoders, you can explore:

1. **Variational Autoencoders (VAEs):**
   - Generate new data samples
   - More probabilistic approach

2. **Denoising Autoencoders:**
   - Remove noise from data
   - Image enhancement

3. **LSTM Autoencoders:**
   - For time-series data
   - Sequential pattern learning

4. **Generative Adversarial Networks (GANs):**
   - Another approach to anomaly detection
   - Can generate realistic fake data

5. **Deep SVDD:**
   - Combines deep learning with SVM
   - State-of-the-art anomaly detection

The autoencoder you built here demonstrates the power of self-supervised learning. The network learns meaningful representations without explicit labels - a key concept in modern AI!

Keep exploring and building! 🎉🔬

