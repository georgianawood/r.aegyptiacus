### Modeling Contextual Influences on Vocal Communication in Egyptian Fruit Bat

Authors: Georgiana Wood, Nicolas Anderson, Ian Shuman

Last Update: 04/28/2024

GitHub repository: https://github.com/georgianawood/r.aegyptiacus

Data: Prat, Yosef; Taub, Mor; Pratt, Ester; Yovel, Yossi (2017). An annotated dataset of Egyptian fruit bat vocalizations across varying contexts and during vocal ontogeny. figshare. Collection. https://doi.org/10.6084/m9.figshare.c.3666502.v2

---
**Objectives:**
This project aims to develop a machine learning model that integrates audio data and contextual information to understand the influence of social context on vocal communication dynamics between pairs of Egyptian fruit bats.

## **Pipeline**


# 1) Identify pairs most frequently communicating

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

The audio was downloaded for individual 215 talking to all addressees, 215 to 207, and 207 to 215.

*Opportunity for future research: Looking into more pairs, male vs. female, adult vs. pup, etc*

## 1.2) Downloading bulk audio files (.WAV) from figshare

Using the ```FileInfo.csv``` file from figshare, we can gather information about which audio files correlate to the interactions we are looking for. After merging ```FileInfo.csv``` with our subsetted dataframe with only individual 207 & 215 (or 215 to all / 215 to 207), we now know what files to obtain.

```
import pandas as pd

df = pd.read_csv('../student practicum/Annotations.csv')

df['Emitter'] = df['Emitter'].abs()
df['Addressee'] = df['Addressee'].abs()

# remove 0, 11, and 12 from context (unknown, unspecified, and sleeping)
df = df[~df['Context'].isin([0, 11, 12])]

# subset dataframe
subset_df = df[df['Emitter'] == 215]
subset_df.to_csv('subset_file.csv', index=False)
```

```
filt = pd.read_csv('subset_file.csv')
info = pd.read_csv('FileInfo.csv', on_bad_lines='skip')

merged = pd.merge(filt, info, on="FileID")
merged.to_csv('merged.csv')
```


From figshare, the files ```files201, files202, files203, files 204, files205, files206, files207, files208, files209, files210, files211, files212, files213, files214, files215, files216, files217, files218, files219, files220, files221, files222, files223, files224``` have been downloaded to my local operating system. From here, the zipped ```files*``` folders are uploaded to SeaWulf, Stony Brook University's HPC system. 
 
 *Warning: even as zipped files, this is over 80 GB of data so proceed with caution if downloading locally.*

# 2) Isolation of audio files

Now in SeaWulf (using unlimited storage!) we must identify and isolate only the audio files that will be used for the project. Using this python script, we are left with only the .WAV files needed to proceed.

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

bigfiles = ['201', '202', '203', '204', '205', '206', '207', '208', '209', '210','211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224']

for file in bigfiles:
        with zipfile.ZipFile('/gpfs/scratch/gnwood/dcs_521/files'+file+'.zip', 'r') as zip_ref:
                zip_ref.extractall('/gpfs/scratch/gnwood/dcs_521/audio_files')

        for filenames in os.listdir('/gpfs/scratch/gnwood/dcs_521/audio_files'):
                if filenames not in list(filename):
                        os.remove('/gpfs/scratch/gnwood/dcs_521/audio_files/'+filenames)

        print(f"File {file} completed")
```

Data in ```/gpfs/scratch/gnwood/dcs_521/audiofiles``` is ready to be used for machine learning applications.

# 3) Preliminary Data Visualization

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


# Display the Audio objects so you can listen to them
for audio in audios:
    display(audio)

#Visualize the data
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
# 4.1) Transform audio files into predictor variables using MFCCs and padding 
Needed to transform audio data and standardize audio data length for model implemenation.
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
    
# Path to directory containing audio files
dir_path = 'audio_files' #can change to audio_file-2 or audio_files-3 depending on the desired dataset

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

    #y, sr = librosa.load(audio_file, sr=None)

y:  Each element of this array represents the amplitude of the audio signal at a particular point in time.

sr: This variable holds the sampling rate of the audio signal, which represents the number of samples (or data points) per second and the frequency range of the audio signal that can be accurately represented.

**Extracting Mel-frequency cepstral coefficients (MFCCs)**

    #extract_mfcc(audio_file, num_mfcc=13):

By transforming the audio data using Mel-frequency cepstral coefficients (MFCCs), we are able to capture multiple characteristics of the audio data that can then be used in machine learning algorithms. 




**Determine maximum length for padding**

    #max_length = 0
    #for wav_file in wav_files:
        #mfccs = extract_mfcc(wav_file)
        #max_length = max(max_length, mfccs.shape[1])

Most machine learning algorithms typically require data inputs of equal length. By padding shorter audio files with MFCCs equal to 0, this ensures that each audio file has the same shape of features.
 

---
# 4.2) Random Forests
Implemented random forests as a classification tool, which outputs the result of multiple decision trees to reach a single result.

## 4.2.1) Random Forest Classifier

```
#Code for extracting the contexts of each .WAV file for use as the response variable once we have the actual data
file_names = [s[10:] for s in wav_files] #get rid of the attached directory
file_names

#load in the merged csv
import pandas as pd
file_path = "merged.csv" #change to merged-2.csv or merged-3.csv depending on the dataset being used
df = pd.read_csv(file_path)
df

#test = ['121219002026838785.WAV', '121219021239831920.WAV'] test data for making sure the extraction works

#perform the extraction
contexts = []
for i in file_names:       #change file_names to test for troubleshooting
    row = df.loc[df['File name'] == i]
    contexts.append(row.at[row.index[0],'Context'])
print(contexts)
y = contexts #this is what we should use for the RandomForest response variable
```


```
import numpy as np
from sklearn import model_selection   
from sklearn import metrics
from sklearn import ensemble

# Setting MFCC feautres and audio contexts
X = mfcc_features_array 
y = y #np.tile(np.arange(1, 6), 5) # dummy data

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

```
def plot_confusion_matrix(classifier, y_test, y_test_pred):
    cm = metrics.confusion_matrix(y_test, y_test_pred, labels=classifier.classes_)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                                          display_labels=classifier.classes_)
    disp.plot()
    
plot_confusion_matrix(classifier, y_test, y_test_pred);
```


**Re-run After Eliminating Contexts 0, 11 and 12**

```
positions = [i for i, x in enumerate(contexts) if x in (0, 1, 11, 12)] #change to filter out contexts you don't want to predict
file_names_filtered = [x for i, x in enumerate(file_names) if x in set(df['File name']) and i not in positions]
contexts_filtered = [x for i, x in enumerate(contexts) if i not in positions]

# Path to directory containing audio files
dir_path = 'audio_files' #change to audio_files-2 or audio_files-3 depending on the dataset being used

# Get a list of all .wav files that do not have context 11 or 12 in the folder
wav_files2 = [os.path.join(dir_path, name) for name in file_names_filtered]

# List to store extracted features
mfcc_features = []

# Determine maximum length for padding
max_length = 0
for wav_file in wav_files2:
    if wav_file[13:] in set(df['File name']):
        mfccs = extract_mfcc(wav_file)
        max_length = max(max_length, mfccs.shape[1])

# Loop through audio files
for wav_file in wav_files2:
    if wav_file[13:] in set(df['File name']):
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

#y = contexts_filtered
```

```
# Setting MFCC feautres and audio contexts
X = mfcc_features_array 
y = y

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

plot_confusion_matrix(classifier, y_test, y_test_pred);
```
## 4.2.2) Random Forest with Random Variable
Implemented a random variable to our random forest model to capture the effect of different individuals on the data. 
```
import os
import librosa
import numpy as np
import pandas as pd

def extract_mfcc(audio_file, num_mfcc=13):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    
    return mfccs

# Path to directory containing audio files
# Change these variable names with the desired files
dir_patha = 'audio_files'
wav_filesa = [os.path.join(dir_patha, f) for f in os.listdir(dir_patha) if f.endswith('.WAV')]

dir_pathb = 'audio_files2'
wav_filesb = [os.path.join(dir_pathb, f) for f in os.listdir(dir_pathb) if f.endswith('.WAV')]

wav_filesall = wav_filesa + wav_filesb

# Assigning context to audio data
file_namesa = [s[12:] for s in wav_filesa]
file_namesb = [s[13:] for s in wav_filesb]
file_namesall = file_namesa + file_namesb

file_patha = "merged.csv"
dfa = pd.read_csv(file_patha)
file_pathb = "merged-2.csv"
dfb = pd.read_csv(file_pathb)

contextsa = []
contextsb = []
for i in file_namesa:    
    row = dfa.loc[dfa['File name'] == i]
    contextsa.append(row.at[row.index[0],'Context'])
for i in file_namesb:  
    row = dfb.loc[dfb['File name'] == i]
    contextsb.append(row.at[row.index[0],'Context'])
contextsall = contextsa + contextsall

positions = [i for i, x in enumerate(contextsall) if x in (0, 11, 12)]
file_names_filtered = [x for i, x in enumerate(file_namesall) if i not in positions]
contexts_filtered = [x for i, x in enumerate(contextsall) if i not in positions]

# Assigning emitter to audio data
emittera = []
emitterb = []
for i in file_namesa:    
    row = dfa.loc[dfa['File name'] == i]
    emittera.append(row.at[row.index[0],'Emitter'])
for i in file_namesb:    
    row = dfb.loc[dfb['File name'] == i]
    emitterb.append(row.at[row.index[0],'Emitter'])
emitterall = emittera + emitterb

# List to store extracted features
mfcc_features = []

# Determine maximum length for padding
max_length = 0
for wav_file in wav_filesall:
    mfccs = extract_mfcc(wav_file)
    max_length = max(max_length, mfccs.shape[1])

# Loop through audio files
for wav_file in wav_filesall:
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
print(mfcc_features_array.shape)
mfcc_features_array.shape[0]

# This trains the random forest classifier and produces the model accuracy and confusion matrix
from sklearn import model_selection   
from sklearn import metrics
from merf import MERF

# Create our variables
num_samples=len(mfcc_features_array[0]) # the number of samples in our data
num_clusters=2 # the number of individuals 

X = mfcc_features_array 
y = np.array(contextsall)

c = np.array(emitterall) 
z = pd.DataFrame(data=1, index=range(mfcc_features_array.shape[0]), columns=['Value'])

# Split data into train and test sets
X_train, X_test, y_train, y_test, z_train, z_test, c_train, c_test = model_selection.train_test_split(X, y, z, c, train_size=0.8, random_state=42)

# Reshape the 3D array into a 2D array as the Random Forest expects a 2D array
X_train_2d = X_train.reshape((X_train.shape[0], -1))
X_test_2d = X_test.reshape((X_test.shape[0], -1))

# Convert cluster arrays to pandas Series, cluster data must be a series
c_train = pd.Series(c_train)
c_test = pd.Series(c_test)

# Initialize and fit MERF model
merf = MERF() 
merf.fit(X_train_2d, z_train, c_train, y_train)

# Predict context on test set
y_pred = merf.predict(X_test_2d, z_test, c_test)
print(y_pred)

# Round the predicted values to the nearest integer,
y_pred_rounded = np.round(y_pred).astype(int)
print(y_pred_rounded)


# Calculate accuracy
accuracy = metrics.accuracy_score(y_test, y_pred_rounded)
print("Accuracy:", accuracy)
print(metrics.confusion_matrix(y_test, y_pred_rounded))


```
Documentation on the MERF module:
https://manifoldai.github.io/merf/

Implementing the MERF module:
https://towardsdatascience.com/mixed-effects-random-forests-6ecbb85cb177

---

## 4.2.3) Variable Importance of MFCCs

```
import numpy as np
from sklearn import model_selection   
from sklearn import metrics
from sklearn import ensemble
from sklearn import svm


# Setting MFCC feautres and audio contexts
X = mfcc_features_array 
y = contexts_filtered #contexts2 if only considering agressiveness

#Splitting the data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size = 0.8)

accuracy = np.empty((0,))
importances = np.empty((0,))
for i in range(0, max_length):
    # Reshape the 3D array into a 2D array
    X_train_2d = X_train[:, :, i]
    X_test_2d = X_test[:, :, i] #take a slice of only the first element in the entire padded list

    # Train the classifier
    classifier = ensemble.RandomForestClassifier(n_estimators=10000) 
    classifier.fit(X_train_2d, y_train)

    # Predict labels for test data
    y_test_pred = classifier.predict(X_test_2d)
    importance = classifier.feature_importances_
    importances = np.append(importances, classifier.feature_importances_)
    accuracy = np.append(accuracy, metrics.accuracy_score(y_test,y_test_pred))

importancesb = importances.reshape((13, -1))
print("The accuracy of the model is", metrics.accuracy_score (y_test,y_test_pred))  
```

```
averages = np.mean(importancesb, axis=1)
averages
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots()
ax.set_title('Mean Decrease in Accuracy for MFCCs Over All Timesteps')

# Create a violin plot for each row
sns.violinplot(data=importancesb.T, ax=ax)

# Show the plot
plt.show()

np.save('/Users/anncrumlish/Downloads/bats/importancesb.npy', importancesb)
```


# 4.3) Other model considerations

When working with audio data represented as Mel-frequency cepstral coefficients (MFCCs) and machine learning alrogrism tasked with classificaion, there are a few other models used for such tasks: Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) Networks, and Support Vector Machine (SVM).

## 4.3.1) Neural Network 
Applied a neural network to audio data from one individual. 

```
import os
import librosa
import numpy as np
import pandas as pd

def extract_mfcc(audio_file, num_mfcc=13):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
    
    return mfccs

# Path to directory containing audio files
# Change these variable names with the desired files
dir_pathb = 'audio_files2'

# Get a list of all .wav files in the folder
wav_filesb = [os.path.join(dir_pathb, f) for f in os.listdir(dir_pathb) if f.endswith('.WAV')]

# Assigning context to audio data
file_namesb = [s[13:] for s in wav_filesb]

file_pathb = "merged-2.csv"
dfb = pd.read_csv(file_pathb)

contextsb = []

for i in file_namesb:    
    row = dfb.loc[dfb['File name'] == i]
    contextsb.append(row.at[row.index[0],'Context'])

positions = [i for i, x in enumerate(contextsb) if x in (0, 11, 12)]
file_names_filtered = [x for i, x in enumerate(file_namesb) if i not in positions]
contexts_filtered = [x for i, x in enumerate(contextsb) if i not in positions]

# List to store extracted features
mfcc_features = []

# Determine maximum length for padding
max_length = 0
for wav_file in wav_filesb:
    mfccs = extract_mfcc(wav_file)
    max_length = max(max_length, mfccs.shape[1])

# Loop through audio files
for wav_file in wav_filesb:
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

#Setting up the neural network
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn import model_selection 


# Setting MFCC feautres and audio contexts
X = mfcc_features_array 
y = np.array(contexts_filtered)

num_samples = mfcc_features_array.shape[0]
num_mfcc_coeffs = mfcc_features_array.shape[1]
num_features = mfcc_features_array.shape[2]
num_contexts = len(contexts_filtered)

# Build the RNN/LSTM model
model = models.Sequential([
    layers.LSTM(64, input_shape=(num_mfcc_coeffs, num_features)),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_contexts, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Splitting data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size = 0.8)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

```

**Building the model**
```
# Build the RNN/LSTM model
#model = models.Sequential([
    #layers.LSTM(64, input_shape=(num_mfcc_coeffs, num_features)),
    #layers.Dense(64, activation='relu'),
    #layers.Dense(num_contexts, activation='softmax')
])
```
We define the architecture of our neural network model using the Sequential API. Inside the sequential model, we stack layers one after the other.
The middle layer allows the model to learn complex patterns and relationships in the data while the final layer implements the softmax activation function, which is suitable for multi-class classification problems.


**Compiling the model**
```
#model.compile(optimizer='adam',
              #loss='sparse_categorical_crossentropy',
              #metrics=['accuracy'])

```
We compile the model, specifying the optimizer, loss function, and metrics for evaluation. Here, we use the Adam optimizer, sparse categorical cross-entropy loss (suitable for integer-encoded class labels), and accuracy as the evaluation metric.



**Training the model**
```
#model.fit(X_train, y_train, epochs=10, batch_size=32)
```
We train the model using the fit method and specify the number of epochs (iterations over the entire dataset), batch size (number of samples per gradient update),

## 4.3.2) Support Vector Machine (SVM)
```
#This code picks up after the data has been processed (same data inputs as for random forest)
import numpy as np
from sklearn import model_selection   
from sklearn import metrics
from sklearn import ensemble
from sklearn import svm


# Setting MFCC feautres and audio contexts
X = mfcc_features_array 
y = contexts_filtered #contexts2 if only considering agressiveness

#Splitting the data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size = 0.8)

accuracy = np.empty((0,))
importances = np.empty((0,))
#for i in range(0, max_length):
# Reshape the 3D array into a 2D array
X_train_2d = X_train.reshape((X_train.shape[0], -1))
   

X_test_2d = X_test.reshape((X_test.shape[0], -1))
    

    # Train the classifier
classifier = svm.SVC(kernel = 'poly', gamma="auto")
classifier.fit(X_train_2d, y_train)

    # Predict labels for test data
y_test_pred = classifier.predict(X_test_2d)
#importance = classifier.feature_importances_
    #print("The accuracy of the model is", metrics.accuracy_score (y_test,y_test_pred))
    #print("The confusion matrix of the model is:")
    #print(metrics.confusion_matrix(y_test, y_test_pred))
    #importances = np.append(importances, classifier.feature_importances_)
    #accuracy = np.append(accuracy, metrics.accuracy_score(y_test,y_test_pred))

#importancesb = importances.reshape((13, -1))
print("The accuracy of the model is", metrics.accuracy_score (y_test,y_test_pred))  

```
# 5) Visualizations

## 5.1) Histogram of Contexts
```
#Change axes/titles depending on the dataset being used
import matplotlib.pyplot as plt
counts, edges, patches = plt.hist(contexts, edgecolor='black')

#plt.hist(contexts, edgecolor='black', bins = 10, align = 'mid')
plt.xlabel('Context')
plt.ylabel('Number of Occurrences')
plt.title('Number of Calls Emitted by Bat 215 in Each Context')
tick_locs = (edges)
tick_labels = ['0', '1','2', '3', '4', '5', '6','7', '8', "9", "10"]
plt.xticks(tick_locs - 0.5, tick_labels)
# Show the plot
plt.show()
```

