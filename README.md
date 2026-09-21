# VAE Music Generation and Reconstruction

## Overview

This repository contains experiments in **music reconstruction and generation using autoencoders and Variational Autoencoders (VAEs)**.

The project works with `.wav` audio files, extracts pitch information using **Aubio**, converts the audio into numerical pitch sequences, and then trains neural-network models to reconstruct or generate similar musical patterns.

Two related notebook implementations are included:

1. `VAE_modified_.ipynb`
   - Implements a **standard dense autoencoder**
   - Uses extracted MIDI-like pitch sequences
   - Learns a compressed latent representation
   - Reconstructs pitch sequences
   - Converts the reconstruction back into audio

2. `VAE_to_generate_similar_music.ipynb`
   - Implements a **Variational Autoencoder**
   - Uses latent variables `z_mean` and `z_log_var`
   - Applies the reparameterization trick
   - Reconstructs fixed-length pitch vectors
   - Evaluates reconstruction error using MSE

> Although the first notebook is named `VAE_modified_.ipynb`, its architecture is technically a conventional autoencoder because it does not use probabilistic latent sampling or KL-divergence regularization.

---

## Project Objective

The primary objective is to explore how latent-space neural networks can learn compressed musical representations from audio pitch sequences.

The project focuses on:

- Extracting pitch information from audio
- Converting audio into machine-learning-friendly numerical representations
- Compressing musical sequences into latent features
- Reconstructing musical pitch patterns
- Exploring generative modeling using a VAE
- Converting generated or reconstructed pitch values back into audio

---

## Dataset

The local dataset is organized into two folders:

```text
DL_dataset/
├── Major/
└── Minor/
```

The audio files consist of `.wav` samples categorized into **Major** and **Minor** groups.

The original audio dataset is intentionally excluded from this GitHub repository because its redistribution rights were not independently verified.

The notebooks expect audio data to be available locally or through Google Drive.

---

## Audio Feature Extraction

Pitch extraction is performed using the **Aubio** library.

The pitch detector uses:

```text
Algorithm: YIN
Sample Rate: 44100 Hz
FFT Window Size: 4096
Hop Size: 512
Pitch Unit: MIDI
```

The process is:

```text
WAV Audio
   │
   ▼
Aubio Source
   │
   ▼
YIN Pitch Detection
   │
   ▼
MIDI-Like Pitch Values
   │
   ▼
Pitch Sequence
```

Each audio file is therefore represented as a sequence of estimated pitch values.

---

# Notebook 1: Dense Autoencoder

## File

```text
VAE_modified_.ipynb
```

This notebook processes multiple `.wav` files from the `Major` and `Minor` folders.

---

## Workflow

```text
Major / Minor WAV Files
          │
          ▼
     Pitch Extraction
          │
          ▼
 Sequence Padding
          │
          ▼
 MIDI Normalization
          │
          ▼
   Train / Test Split
          │
          ▼
    Dense Encoder
          │
          ▼
 32-Dimensional Latent Space
          │
          ▼
    Dense Decoder
          │
          ▼
 Reconstructed Pitch Sequence
          │
          ▼
  Audio Reconstruction
```

---

## Preprocessing

Pitch sequences may contain different numbers of values because the audio files have different lengths.

The notebook therefore uses sequence padding:

```python
pad_sequences(...)
```

so that all samples have the same dimensionality.

The resulting MIDI pitch values are normalized using:

```text
pitch / 127
```

because standard MIDI note values are represented on a 0–127 scale.

---

## Train-Test Split

The normalized pitch dataset is divided using:

```text
80% Training
20% Testing
```

with:

```python
random_state = 42
```

---

## Autoencoder Architecture

The implemented architecture is:

```text
Input Layer
    │
    ▼
Dense Layer
128 Units + ReLU
    │
    ▼
Latent Layer
32 Units + ReLU
    │
    ▼
Dense Layer
128 Units + ReLU
    │
    ▼
Output Layer
Sigmoid
```

The latent space has:

```text
32 dimensions
```

---

## Training Configuration

The autoencoder is compiled using:

```text
Optimizer: Adam
Loss Function: Mean Squared Error
Epochs: 50
Batch Size: 16
```

Training reconstructs the input itself:

```text
Input Pitch Sequence → Autoencoder → Reconstructed Pitch Sequence
```

---

## Audio Reconstruction

After prediction, reconstructed normalized values are converted back to MIDI-style pitch values:

```text
Reconstructed Value × 127
```

The first reconstructed sample is then converted into a waveform using sinusoidal synthesis.

For every pitch:

```text
MIDI Pitch
    ↓
Frequency Conversion
    ↓
Sine Wave
    ↓
Audio Signal
```

The MIDI-to-frequency equation follows:

```text
frequency = 440 × 2^((pitch - 69) / 12)
```

The final audio file is saved as:

```text
reconstructed_output.wav
```

---

# Notebook 2: Variational Autoencoder

## File

```text
VAE_to_generate_similar_music.ipynb
```

This notebook implements a more explicit **Variational Autoencoder architecture**.

It processes extracted pitch sequences into fixed-length vectors.

---

## Fixed Input Representation

Each audio sample is converted into:

```text
300 pitch features
```

Sequences shorter than 300 values are padded.

Sequences longer than 300 values are truncated.

The resulting pitch dataset therefore has the form:

```text
Number of Samples × 300 Features
```

In the notebook experiment, the processed dataset shape is:

```text
23 × 300
```

---

## Pitch Dataset Generation

The notebook extracts pitch values from every `.wav` file and saves them as:

```text
pitch_dataset.csv
```

The process is:

```text
WAV Files
   │
   ▼
Pitch Extraction
   │
   ▼
Padding / Truncation
   │
   ▼
300-Dimensional Vectors
   │
   ▼
pitch_dataset.csv
```

---

# VAE Architecture

## Encoder

The encoder receives:

```text
300 input features
```

and passes them through:

```text
Input
 ↓
Dense 256 + ReLU
 ↓
z_mean
 ↓
z_log_var
```

The latent-space dimension is:

```text
16
```

---

## Reparameterization Trick

A custom sampling layer generates the latent vector using:

```text
z = z_mean + exp(0.5 × z_log_var) × epsilon
```

where:

```text
epsilon ~ Normal Distribution
```

This stochastic latent representation is one of the defining characteristics of a Variational Autoencoder.

---

## Decoder

The decoder architecture is:

```text
Latent Vector (16)
       │
       ▼
Dense Layer
256 Units + ReLU
       │
       ▼
Output Layer
300 Units + Sigmoid
```

The decoder attempts to reconstruct the original pitch representation.

---

## Training

The VAE experiment is trained for:

```text
10 epochs
```

The recorded training loss decreases across epochs, indicating that the model is learning to reconstruct the pitch vectors.

The notebook reports a final reconstruction loss of approximately:

```text
0.038116
```

for the processed dataset evaluation.

---

# Individual Audio Reconstruction

The notebook also evaluates reconstruction on a single audio file.

The original pitch sequence contains:

```text
345 extracted pitch values
```

It is then converted into a fixed:

```text
300-dimensional vector
```

before being passed into the VAE.

For this reconstruction experiment, the notebook reports:

```text
Reconstruction Loss (MSE): 0.034182
```

A reconstructed audio output is then generated from the predicted pitch representation.

---

## Evaluation Metric

The primary evaluation measure is:

### Mean Squared Error

```text
MSE = average((original - reconstructed)²)
```

A lower reconstruction MSE indicates that the generated pitch representation is numerically closer to the original input representation.

The reported single-sample value is:

```text
MSE = 0.034182
```

---

# Major and Minor Audio Classes

The dataset is organized into:

```text
Major
Minor
```

The first notebook stores labels as:

```text
Minor = 0
Major = 1
```

However, these labels are not used for supervised classification in the autoencoder training.

The current model learns to reconstruct pitch sequences rather than classify major versus minor music.

This leaves major/minor classification as a possible future extension.

---

# Experimental Chord Mapping

One notebook also contains an experimental rule-based conversion of numerical pitch ranges to chord names such as:

- C Major
- G Major
- D Major
- A Minor
- E Minor
- F Major
- B Minor
- A Major
- E Major
- D Minor

This section is exploratory and should not be interpreted as a complete music-theory chord-recognition system.

The mapping is based on manually specified numerical pitch intervals.

---

# Autoencoder vs Variational Autoencoder

| Feature | Dense Autoencoder | Variational Autoencoder |
|---|---|---|
| Notebook | `VAE_modified_.ipynb` | `VAE_to_generate_similar_music.ipynb` |
| Input | Padded pitch sequences | 300-value pitch vectors |
| Latent Size | 32 | 16 |
| Probabilistic Latent Space | No | Yes |
| `z_mean` | No | Yes |
| `z_log_var` | No | Yes |
| Reparameterization | No | Yes |
| Reconstruction | Yes | Yes |
| Audio Output | Yes | Yes |
| Training Epochs | 50 | 10 |

---

# Technologies Used

## Programming

- Python
- Jupyter Notebook
- Google Colab

## Deep Learning

- TensorFlow
- Keras
- Autoencoders
- Variational Autoencoders

## Audio Processing

- Aubio
- SciPy
- SoundFile

## Data Processing

- NumPy
- Pandas
- Scikit-learn

---

# Repository Structure

```text
VAE-Music-Generation/
│
├── VAE_modified_.ipynb
│   └── Dense autoencoder for pitch reconstruction
│
├── VAE_to_generate_similar_music.ipynb
│   └── Variational Autoencoder experiment
│
├── reconstructed_output.wav
│   └── Reconstructed audio generated by the model
│
├── Reinforcement Learning.key
│   └── Supporting project presentation/material
│
├── .gitignore
│
└── README.md
```

The original:

```text
DL_dataset/
```

folder is intentionally excluded from the repository.

---

# Key Findings

The experiments demonstrate that:

- WAV files can be converted into numerical pitch sequences using Aubio.
- Autoencoders can learn compressed representations of musical pitch sequences.
- A 32-dimensional deterministic latent representation can reconstruct padded audio pitch data.
- A Variational Autoencoder can model pitch information using a stochastic latent space.
- The VAE experiment uses a 16-dimensional latent representation.
- Pitch vectors can be decoded and reconstructed into audio.
- The VAE experiment achieved an MSE of approximately **0.034182** for the evaluated individual reconstruction.
- The generated waveform provides a simplified reconstruction of the learned pitch information.

---

# Limitations

## Pitch-Only Representation

The models primarily represent music using estimated pitch.

They do not fully model:

- Timbre
- Instrument characteristics
- Dynamics
- Articulation
- Rhythm complexity
- Harmonic structure
- Expressive performance information

As a result, reconstructed audio is a simplified representation of the original music.

---

## Synthetic Sine-Wave Reconstruction

The first notebook reconstructs audio using sine waves generated from MIDI-style pitches.

This preserves approximate pitch information but does not recreate the original instrument sound.

---

## Small VAE Dataset

The VAE notebook processes only:

```text
23 samples
```

in the recorded experiment.

This dataset is too small to support strong claims about general music-generation performance.

A significantly larger dataset would be required for robust generative modeling.

---

## Experimental Chord Mapping

The chord mapping section uses manually defined pitch ranges.

True chord recognition normally requires analysis of multiple simultaneous notes, harmonic relationships, and temporal context.

---

## Fixed-Length Input

The VAE truncates or pads pitch sequences to 300 features.

This can cause information loss for longer recordings and introduce padding for shorter recordings.

Sequence-based architectures could provide a better approach for variable-length music.

---

# Future Work

Possible future extensions include:

- Larger music datasets
- MIDI-native datasets
- Conditional VAEs
- Major/minor conditioned generation
- Genre-conditioned generation
- Recurrent VAEs
- LSTM-based decoders
- Transformer-based music generation
- Convolutional VAEs using spectrograms
- Mel-spectrogram representations
- Waveform-level generative models
- Better chord detection
- Rhythm modeling
- Tempo modeling
- Timbre reconstruction
- Instrument-conditioned generation
- Latent-space interpolation
- Music similarity analysis
- Objective audio-quality metrics
- Human listening evaluation

---

# Potential Latent-Space Applications

A properly trained VAE could allow experiments such as:

```text
Song A Latent Vector
        +
Interpolation
        +
Song B Latent Vector
        ↓
New Musical Representation
```

This could enable exploration of smooth transitions between learned musical patterns.

---

# Academic Context

This repository is maintained for **academic and experimental deep-learning purposes**.

The project explores the use of latent-variable neural networks for music reconstruction and generative modeling.

It is intended as a learning-oriented implementation rather than a production-grade music-generation system.

---

# Dataset Notice

The original audio dataset is not redistributed in this repository.

The dataset folder is intentionally excluded because the source and redistribution license of the audio files were not independently verified.

Users wishing to reproduce the project should use audio files for which they have appropriate usage rights.

---

# Disclaimer

The reconstructed audio is generated from learned or processed pitch representations and should not be interpreted as a faithful reproduction of the original musical recording.

Any external audio used with this project should comply with its applicable copyright and licensing terms.
