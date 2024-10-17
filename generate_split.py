import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = 'data/mapping/final_dataset.csv'  # Update this path
df = pd.read_csv(file_path)

# Group the data by the 'name' field
grouped = df.groupby('name')

# Prepare lists to hold the data splits
train_list = []
val_list = []
test_list = []

# Improved logic for splitting small groups
for name, group in grouped:
    if len(group) == 1:
        # Assign the single sample to the training set
        train = group
        val, test = pd.DataFrame(), pd.DataFrame()
    elif len(group) == 2:
        # Assign one sample to train and the other to val/test
        train, val = group.iloc[:1], group.iloc[1:]
        test = pd.DataFrame()
    elif len(group) == 3:
        # Assign one sample to each split
        train, val, test = group.iloc[0:1], group.iloc[1:2], group.iloc[2:]
    else:
        # Standard splitting for larger groups
        train, temp = train_test_split(group, test_size=0.3, random_state=42)
        val, test = train_test_split(temp, test_size=0.5, random_state=42)
    
    train_list.append(train)
    val_list.append(val)
    test_list.append(test)

# Concatenate the splits back into full DataFrames
train_df = pd.concat(train_list).reset_index(drop=True)
val_df = pd.concat(val_list).reset_index(drop=True)
test_df = pd.concat(test_list).reset_index(drop=True)

# Display the number of samples in each set
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# Save the splits to separate CSV files
train_df.to_csv('train_dataset.csv', index=False)
val_df.to_csv('val_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

print("Data splits saved to 'train_dataset.csv', 'val_dataset.csv', and 'test_dataset.csv'")