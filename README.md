### Modeling Contextual Influences on Vocal Communication in Egyptian Fruit
Authors: Georgiana Wood, Nicolas Anderson, Ian Shuman

Last Update: 04/10/2024

GitHub repository: https://github.com/georgianawood/r.aegyptiacus

Data: Prat, Yosef; Taub, Mor; Pratt, Ester; Yovel, Yossi (2017). An annotated dataset of Egyptian fruit bat vocalizations across varying contexts and during vocal ontogeny. figshare. Collection. https://doi.org/10.6084/m9.figshare.c.3666502.v2

---
**Objectives:**
This project aims to develop a machine learning model that integrates audio data and contextual information to understand the influence of social context on vocal communication dynamics between pairs of Egyptian fruit bats.

## **Pipeline**


**1) Identify pairs most frequently communicating**

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

**2) Downloading bulk audio files (.WAV) from figshare**

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

**3) Isolation of audio files**

Now in SeaWulf (using unlimited storage!) we must identify and isolate only the 660 audio files that will be used for the project. Using this python script, we are left with only the .WAV files needed to proceed.

```
module load anaconda/3

import pandas as pd
import zipfile

df = pd.read_csv('/gpfs/scratch/gnwood/dcs_521/csv/merged.csv')

filename = df['File name']

bigfiles = ['210','211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '224'] 

for file in bigfiles:
        with zipfile.ZipFile('/gpfs/scratch/gnwood/dcs_521/file'+file+'.zip', 'r') as zip_ref:
                zip_ref.extractall('/gpfs/scratch/gnwood/dcs_521/audiofiles')

for filenames in os.listdir('/gpfs/scratch/gnwood/dcs_521/audiofiles'):
        if filenames not in list(filename):
                os.remove('/gpfs/scratch/gnwood/dcs_521/audiofiles/'+filenames)
```

Data in ```/gpfs/scratch/gnwood/dcs_521/audiofiles``` is ready to be used for machine learning applications.


---
