### Modeling Contextual Influences on Vocal Communication in Egyptian Fruit Bat

Authors: Georgiana Wood, Nicolas Anderson, Ian Shuman

Last Update: 04/10/2024

GitHub repository: https://github.com/georgianawood/r.aegyptiacus

Data: Prat, Yosef; Taub, Mor; Pratt, Ester; Yovel, Yossi (2017). An annotated dataset of Egyptian fruit bat vocalizations across varying contexts and during vocal ontogeny. figshare. Collection. https://doi.org/10.6084/m9.figshare.c.3666502.v2

---
**Objectives:**
This project aims to develop a machine learning model that integrates audio data and contextual information to understand the influence of social context on vocal communication dynamics between pairs of Egyptian fruit bats.

## **Pipeline**


## 1) Identify pairs most frequently communicating

From our Annotations.csv file we are looking to figure out which pairs of bats are interacting the most.

```
import pandas as pd

data = pd.read_csv('Annotations.csv')
emit = 'Emitter'
address = 'Addressee'

# Absolute values (negative numbers indicate individual is either emitter or addressee)
data[emit] = data[emit].abs()
data[address] = data[address].abs()

# Remove "0" from the data (0 indicates unknown individual)
data = data[(data[emit] != 0) & (data[address] != 0)]

# Count repeat interactions
repeat_interactions = data.groupby([emit, address]).size().reset_index(name='count').sort_values(by='count', ascending=False)

repeat_interactions = repeat_interactions[repeat_interactions[emit] != repeat_interactions[address]]

print(repeat_interactions)
```

After running this code, we found out that individual 215 and individual 207 are interacting the most out of the whole dataset (5,588 times). 215 is a female pup and 207 is a male pup, they were born two days apart. For the remainder of this project, we will be focusing on these two individuals. 

*Opportunity for future research: Looking into more pairs, male vs. female, adult vs. pup, etc*

## 2) Downloading bulk audio files (.WAV) from figshare

Using the ```FileInfo.csv``` file from figshare, we can gather information about which audio files correlate to the interactions we are looking for. After merging ```FileInfo.csv``` with our subsetted dataframe with only individual 207 & 215, we now know what files to obtain.

```
import pandas as pd

filt = pd.read_csv('filtered_data_for_weka.csv')
info = pd.read_csv('FileInfo.csv', on_bad_lines='skip')

merged = pd.merge(filt, info, on="FileID")
merged.to_csv('merged.csv')

set(merged['File folder'])
```


From figshare, the files ```{'files210',
 'files211',
 'files212',
 'files213',
 'files214',
 'files215',
 'files216',
 'files217',
 'files218',
 'files219',
 'files220',
 'files221',
 'files224'}``` have been downloaded to my local operating system. From here, the zipped ```files*``` folders are uploaded to SeaWulf, Stony Brook University's HPC system. 
 
 *Warning: even as zipped files, this is nearly 40 GB of data so proceed with caution if downloading locally.*

## 3.1) Isolation of audio files

Now in SeaWulf (using unlimited storage!) we must identify and isolate only the 660 audio files that will be used for the project. Using this python script, we are left with only the .WAV files needed to proceed.

Load conda modules as well as the custom environment made to run python scripts

```
module load anaconda/3
source activate (ml)
```
```
import pandas as pd
import zipfile
import os

df = pd.read_csv('/gpfs/scratch/gnwood/dcs_521/csv/merged.csv')

filename = df['File name']

bigfiles = ['210','211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '224']

for file in bigfiles:
        with zipfile.ZipFile('/gpfs/scratch/gnwood/dcs_521/files'+file+'.zip', 'r') as zip_ref:
                zip_ref.extractall('/gpfs/scratch/gnwood/dcs_521/audio_files')

        for filenames in os.listdir('/gpfs/scratch/gnwood/dcs_521/audio_files'):
                if filenames not in list(filename):
                        os.remove('/gpfs/scratch/gnwood/dcs_521/audio_files/'+filenames)

        print(f"File {file} completed")
```

Data in ```/gpfs/scratch/gnwood/dcs_521/audiofiles``` is ready to be used for machine learning applications.

## 3.2) Preliminary Data Visualization

Code for loading, playing, and visualizing audio files in Python
```
import os
import wave
import numpy as np
from IPython.display import Audio

# dir_path is the path to a folder of all the .WAV files that we want to access
dir_path = 'test data'

# Get a list of all .wav files in the folder
wav_files = [f for f in os.listdir(dir_path) if f.endswith('.WAV')]

# Create a list to store the Audio objects
audios = []

# Open each .WAV file
for wav_file in wav_files:
    file_path = os.path.join(dir_path, wav_file)
    with wave.open(file_path, 'rb') as wav_file:
        # Read the audio data and convert it to a NumPy array
        nframes = wav_file.getnframes()
        data = np.frombuffer(wav_file.readframes(nframes), dtype=np.int16)

        # Get the frame rate
        rate = wav_file.getframerate()

    # Create an Audio object with the audio data and frame rate
    audio = Audio(data, rate=rate)
    
    # Add the Audio object to the list
    audios.append(audio)

```

```
# Display the Audio objects so you can listen to them
for audio in audios:
    display(audio)
```

```
#Visualize
#You need to install the librosa library onto your computer in order to access the following packages, 
#so type the following command into your terminal: 
#pip install librosa
import librosa
import librosa.display
import matplotlib.pyplot as plt
for wav_file in wav_files:
    file_path = os.path.join(dir_path, wav_file)
    data,sample_rate = librosa.load(file_path)
    plt.figure(figsize=(12,4))
    librosa.display.waveshow(data,sr=sample_rate)
```
---
## 4.1) Transform audio files into predictor variables using MFCCs and padding 

```
import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_mfcc(audio_file, num_mfcc=13):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    
    return mfccs

def scale_features(features): 
    # Initialize StandardScaler
    scaler = StandardScaler()
    
    # Fit scaler to features and transform
    scaled_features = scaler.fit_transform(features.T).T
    
    return scaled_features

# Path to directory containing audio files
dir_path = 'test data'

# Get a list of all .wav files in the folder
wav_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.WAV')]

# List to store extracted features
mfcc_features = []

# Determine maximum length for padding
max_length = 0
for wav_file in wav_files:
    mfccs = extract_mfcc(wav_file)
    max_length = max(max_length, mfccs.shape[1])

# Loop through audio files
for wav_file in wav_files:
    # Extract MFCC features
    mfccs = extract_mfcc(wav_file)
    
    # Pad or truncate MFCC features to maximum length
    num_pad_columns = max_length - mfccs.shape[1]
    if num_pad_columns > 0:
        # Pad the features with zeros
        mfccs = np.pad(mfccs, ((0, 0), (0, num_pad_columns)), mode='constant')
    
    # Append features to list
    mfcc_features.append(mfccs)

# Convert list to numpy array
mfcc_features_array = np.array(mfcc_features)

# Output the shape of the features array
print("Shape of features array:", mfcc_features_array.shape)
```

**Load audio file**

    y, sr = librosa.load(audio_file, sr=None)

y: This variable holds the audio signal as a one-dimensional NumPy array. Each element of this array represents the amplitude of the audio signal at a particular point in time.

sr: This variable holds the sampling rate of the audio signal, which represents the number of samples (or data points) per second. It is an integer value, typically measured in Hz (samples per second). The sampling rate determines the frequency range of the audio signal that can be accurately represented.

**Extracting MFCCs**

    extract_mfcc(audio_file, num_mfcc=13):

num_mfcc: This parameter specifies the number of MFCC coefficients to be extracted. By default, it is set to 13.
Now, why is 13 chosen as the number of MFCC coefficients? There's no hard and fast rule for choosing the number of MFCC coefficients, but traditionally, 13 is a common choice. Here's why:

- Dimensionality Reduction: The purpose of MFCCs is to capture the essential characteristics of the audio signal while reducing its dimensionality. Thirteen coefficients strike a balance between capturing enough information about the spectral shape of the signal and keeping the computational cost manageable.

- Human Auditory System: The human auditory system is believed to process sound in a way that makes it sensitive to certain frequency bands. By choosing 13 MFCC coefficients, we're aiming to capture the most relevant information in a manner that aligns with human perception.

- Historical Convention: The use of 13 MFCC coefficients has been a convention in speech and audio processing for a long time, and many algorithms, libraries, and research papers use this default value.

However, the choice of the number of MFCC coefficients can vary depending on the specific application and the characteristics of the audio data. In some cases, you may choose to use more or fewer coefficients based on experimentation and optimization for your particular task.

*NOTE: When the function is used to loops over audio files, it will produce a 2 dimensional array consisting of the number of MFCC coefficients and the number of MFCC features in each audio file.*

**Fit scaler to features and transform**

    scaled_features = scaler.fit_transform(features.T).T

scaler.fit_transform(features.T): 
The fit_transform() method of the scaler object (scaler) first fits the scaler to the features (features.T) and then transforms the features using the fitted scaler.

features.T is used here because typically the fit_transform() method expects features to be in the shape (n_samples, n_features), where n_samples is the number of samples (e.g., audio files) and n_features is the number of features (e.g., MFCC coefficients). However, in this case, the features are likely in the shape (n_features, n_samples) (each column represents the features of one sample), so features.T is used to transpose the matrix.
.T: The .T at the end of the line transposes the transformed features back to their original shape. This is necessary because fit_transform() returns the transformed features as a transposed matrix, and we want to return them to their original shape.

Overall, this line of code fits the scaler to the features and then scales the features using the fitted scaler, ensuring that each feature is scaled appropriately across the samples.

**Determine maximum length for padding**

    max_length = 0
    for wav_file in wav_files:
        mfccs = extract_mfcc(wav_file)
        max_length = max(max_length, mfccs.shape[1])

max_length = max(max_length, mfccs.shape[1]): After extracting the MFCC features, we calculate the number of coefficients (or features) in the MFCCs using mfccs.shape[1]. The shape of mfccs is a tuple (num_mfcc, num_frames), where num_mfcc is the number of MFCC coefficients and num_frames is the number of frames (or time steps) in the audio. We're interested in the second dimension (num_frames) to determine the length of MFCC features. We update max_length to the maximum value between its current value and the length of MFCC features from the current audio file. This ensures that max_length always stores the maximum length of MFCC features among all audio files processed so far.

In summary, this code snippet iterates over all audio files, extracts the MFCC features from each file, and updates max_length to store the maximum length of MFCC features found among all files. This maximum length is then used for padding the MFCC features to ensure that they all have the same length when converting them into a numpy array. Padding ensures that each audio file has the same shape of features, which is often necessary for processing with machine learning algorithms.

*NOTE: The final output, a NumPy array consisting of all looped audio files, will be a 3 dimensional array detailing the number of audio files, the number of MFCC coefficients, and the number of MFCC features. Because the audio files are padded, the number of MFCC features will be the same across audio files and equal to the that of the file with the largest number of MFCC features.*

---
## 4.2) Random Forest Classifier

```
import numpy as np
from sklearn import model_selection   
from sklearn import metrics
from sklearn import ensemble

# Setting MFCC feautres and audio contexts
X = mfcc_features_array 
y = np.tile(np.arange(1, 6), 5) # dummy data

#Splitting the data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size = 0.8)

# Reshape the 3D array into a 2D array
X_train_2d = X_train.reshape((X_train.shape[0], -1))
X_test_2d = X_test.reshape((X_test.shape[0], -1))

# Train the classifier
classifier = ensemble.RandomForestClassifier(n_estimators=10000) 
classifier.fit(X_train_2d, y_train)

# Predict labels for test data
y_test_pred = classifier.predict(X_test_2d)

print("The accuracy of the model is", metrics.accuracy_score (y_test,y_test_pred))
print("The confusion matrix of the model is:")
print(metrics.confusion_matrix(y_test, y_test_pred))
```
**Setting up MFCC features array and audio contexts**

    X = mfcc_features_array 
    
The number of dimensions will have to be reduced to fit the random forest classifier. In the model, we will need a tuple that corresponds to the number of samples and the number of coefficients by the number of MFCC features. In this case, the dimensions for the model will be the first dimension of the mfcc_features_array, while the second dimension of the model will be the second dimension by third dimension of the mfcc_features_array.

    y = np.tile(np.arange(1, 6), 5) 
    
This must be a list of all of the contexts associated with each file. As of now, this is **dummy data** that simulates 5 contexts, each represented 5 times.

**Dimension verification**

You can print out the array size of X and y using the below to verify the number of dimensions that you have.

    print(X.shape)
    print(y.shape)

**Training the data**

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size = 0.8)
    print("The size of X_train is",X_train.shape)
    
Setting the train size to 0.8 reflects the percentage of data dedicated to training your model. 

**Reshape the 3D array into a 2D array**

    X_train_2d = X_train.reshape((X_train.shape[0], -1))
    X_test_2d = X_test.reshape((X_test.shape[0], -1))
    
This turns the 3D array into a 2D array, whereby the first dimension stays the same size, i.e. the number of samples, while the second dimension is converted into a dimension that is equal to the product of the number of coefficients and number of features. This means that each column now represents the value of a specificed coefficient at a specific point in time. 

**Train the classifier**

    classifier = ensemble.RandomForestClassifier(n_estimators=10000) 
    classifier.fit(X_train_2d, y_train)
    
Setting the number of estimators determines the number of trees that are created. Increasing these estimators increases computation time but increases accuracy. 

---

## 4.3) Other model considertions

When working with audio data represented as Mel-frequency cepstral coefficients (MFCCs) in a 3-dimensional array, you're essentially dealing with a form of time-series data where each sample is characterized by a set of features (MFCC coefficients). Here are a few models commonly used for such tasks:

- Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) Networks: RNNs and their variant LSTMs are well-suited for sequential data like audio. They can learn patterns over time and make predictions based on past inputs. You can feed the MFCCs into the RNN/LSTM layer and use a fully connected layer for classification.

- Convolutional Neural Networks (CNNs): CNNs are often used for image data, but they can also be adapted for 1D data like audio. You can treat each MFCC coefficient as a "channel" and apply 1D convolutions over time. CNNs can capture local patterns effectively.

- Convolutional Recurrent Neural Networks (CRNNs): This is a combination of CNNs and RNNs, where CNNs are used for feature extraction and RNNs (often LSTMs) are used for sequence modeling. CRNNs have shown good performance in tasks involving sequential data like audio.

- Transformer Models: Transformer models, such as the Transformer architecture used in models like BERT, have shown remarkable performance in sequence-to-sequence tasks. They don't rely on recurrence, making them parallelizable and potentially faster to train.

- Hybrid Architectures: You can also experiment with hybrid architectures combining elements of the above models. For example, you could use a CNN for initial feature extraction, followed by an RNN or Transformer for sequence modeling and prediction.

Before deciding on a model, it's crucial to consider factors like the size of your dataset, computational resources available for training, and the complexity of the problem you're trying to solve. You may need to experiment with different architectures and hyperparameters to find the best-performing model for your specific task. Additionally, techniques such as data augmentation and regularization can help improve model generalization.

---
## 5) Visualizations

Something I had questions about myself and maybe some answers regarding MFCs

https://medium.com/@tanveer9812/mfccs-made-easy-7ef383006040

https://dsp.stackexchange.com/questions/38830/whats-the-correct-graphical-interpretation-of-a-series-of-mfcc-vectors

---

