# To  handle class imbalance in a binary classification dataset using numerical data augmentation techniques, specifically SMOTE and Borderline-SMOTE.
# This code was run on the google colab, and the dataset uploaded there

from google.colab import files
uploaded = files.upload()

import pandas as pd
df = pd.read_csv('Imbalanced_data.csv', header = None)
df.head()
df.columns = ['X', 'Y', 'class'] # Rename the Colum of Dataset
df.head()
df['class'].value_counts() # it shows the count of data distributuoin
df['class'].value_counts(normalize=True)*100   # it shows the count of data distributuoin in percentage

#That portion displaying the existing dataset frequency and dataset class 0 and 1
!pip install imbalanced-learn
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
import matplotlib.pyplot as plt
df['class'].value_counts(normalize=True).plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Imbalanced Data Graph')
plt.show()

# Display the imbalance data
X = df[['X', 'Y']]
Y = df['class']

plt.figure()
plt.scatter(X[Y == 0]['X'], X[Y == 0]['Y'], label='Class 0 ->Majority', alpha=0.6)
plt.scatter(X[Y == 1]['X'], X[Y == 1]['Y'], label='Class 1 ->Minority', alpha=0.6)
plt.title("Original Imbalanced Dataset")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

smote = SMOTE(random_state=42)
x_smote, y_smote = smote.fit_resample(X,Y)
y_smote.value_counts()
type(x_smote), getattr(x_smote, "columns", None)


# Draw the plot of distribution after applying SMOTE to the Imbalaced data set


plt.figure()
plt.scatter(x_smote.loc[y_smote ==0, 'X'], x_smote.loc[y_smote ==1, 'Y'], label='Class 0', alpha=0.6)
plt.scatter(x_smote.loc[y_smote ==1, 'X'], x_smote.loc[y_smote ==1, 'Y'], label = 'Class 1 ', alpha=0.6)
plt.title("SMOTE Oversampling")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

# B-SMOTE Implementation
bsmote = BorderlineSMOTE(random_state=42)
x_bsmote, y_bsmote = bsmote.fit_resample(X,Y)
y_bsmote.value_counts()

# Draw the plot of distribution after applying B-SMOTE (BorderlineSMOTE) to the Imbalaced data set

plt.figure()
plt.scatter(x_bsmote.loc[y_bsmote ==0, 'X'], x_bsmote.loc[y_bsmote ==0, 'Y'], label='Class 0', alpha=0.6)
plt.scatter(x_bsmote.loc[y_bsmote ==1, 'X'], x_bsmote.loc[y_bsmote ==1, 'Y'], label = 'Class 1 ', alpha=0.6)
plt.title('B-SMOTE Oversampling')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()