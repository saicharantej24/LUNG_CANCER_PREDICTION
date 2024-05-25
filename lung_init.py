from sklearn.model_selection import train_test_split
import pandas as pd
# Read the CSV file
df = pd.read_csv(r'C:\Users\saich\OneDrive\Desktop\lungcancer.csv')
print("printing columns in dataset:")
print(df.columns)
print("Chceking the datatypes of columns")
print(df.dtypes)
print("Checking whether there is any null values:")
print(df.isnull().sum())
print("Shape of dataset")
print(df.shape)
print(df.head())
print("converting categorical data in to numeric data:")
df['GENDER'] = df['GENDER'].map({'M': 1, 'F': 2})
print(df.head())
df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
print(df.head())    
print(df['LUNG_CANCER'].value_counts())
print("Handling imbalanced dataset")
y = df['LUNG_CANCER']
X = df.drop(['LUNG_CANCER'], axis = 1)
from imblearn.over_sampling import RandomOverSampler

oversampler = RandomOverSampler()
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Concatenate X_resampled and y_resampled into a new dataframe
demo = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=["LUNG_CANCER"])], axis=1)

# Check the class distribution after oversampling
print(demo["LUNG_CANCER"].value_counts())
#print("Standardizing the data")
#from sklearn.preprocessing import StandardScaler
#standardScaler = StandardScaler()
#columns_to_scale = ['AGE']
#demo[columns_to_scale] = standardScaler.fit_transform(demo[columns_to_scale])
#demo.to_csv(r'C:\Users\saich\OneDrive\Desktop\lung\lungcancerfile.csv', index=False)
demo.to_csv(r'C:\Users\saich\OneDrive\Desktop\lungcancer.csv', index=False)



df=pd.read_csv(r'C:\Users\saich\OneDrive\Desktop\lungcancer.csv')
y = df['LUNG_CANCER']
X = df.drop(['LUNG_CANCER'], axis = 1)


from sklearn.preprocessing import MinMaxScaler


# Drop the target variable before scaling
X_scaled = demo.drop(columns=["LUNG_CANCER"])

# Perform Min-Max scaling
scaler_minmax = MinMaxScaler()
X_scaled_minmax = scaler_minmax.fit_transform(X_scaled)

# Convert the scaled array back to DataFrame
X_scaled_minmax_df = pd.DataFrame(X_scaled_minmax, columns=X_scaled.columns)

# Combine scaled features and target into a single DataFrame
demo= pd.concat([X_scaled_minmax_df, y], axis=1)
demo.to_csv(r'C:\Users\saich\OneDrive\Desktop\lungcancerfile.csv', index=False)