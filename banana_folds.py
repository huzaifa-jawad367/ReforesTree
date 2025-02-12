
import pandas as pd
from sklearn.model_selection import KFold

def create_systematic_splits(
    csv_path: str,
    n_folds: int = 5,
    val_ratio: float = 0.2,
    output_prefix: str = "fold"
):
    """
    Create systematic (non-random) k-fold splits of a dataset into
    train, validation, and test CSVs. Each fold's test set is disjoint,
    and the training set is further split to form a validation set.

    Parameters:
    -----------
    csv_path : str
        Path to the input metadata CSV file.
    n_folds : int, optional
        Number of folds (splits). Common choices are 5 or 10. Default is 5.
    val_ratio : float, optional
        Fraction of the training portion to be used as validation. Default is 0.2 (20%).
    output_prefix : str, optional
        Prefix to use for the output CSV filenames. Default is "fold".

    Output:
    -------
    For each fold i in [1..n_folds], three CSVs are created:
    - {output_prefix}_train_i.csv
    - {output_prefix}_val_i.csv
    - {output_prefix}_test_i.csv
    """
    # 1. Read the CSV file
    df = pd.read_csv(csv_path)

    # 2. Initialize the KFold splitter
    #    shuffle=False ensures systematic (non-random) splitting
    kf = KFold(n_splits=n_folds, shuffle=False)

    # 3. Iterate through each fold
    for fold, (train_index, test_index) in enumerate(kf.split(df), start=1):
        # 3a. Split into train and test
        train_df = df.iloc[train_index].reset_index(drop=True)
        test_df  = df.iloc[test_index].reset_index(drop=True)

        # 3b. Further split the train_df into train & validation
        val_size = int(len(train_df) * val_ratio)
        val_df = train_df.iloc[:val_size].reset_index(drop=True)
        final_train_df = train_df.iloc[val_size:].reset_index(drop=True)

        # 4. Save to CSV
        train_csv = f"{output_prefix}_train_{fold}.csv"
        val_csv   = f"{output_prefix}_val_{fold}.csv"
        test_csv  = f"{output_prefix}_test_{fold}.csv"

        final_train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)

        print(f"Fold {fold} created:")
        print(f"  Train: {train_csv}")
        print(f"  Val:   {val_csv}")
        print(f"  Test:  {test_csv}\n")

# --- Usage Example ---
if __name__ == "__main__":
    create_systematic_splits(
        csv_path="/content/data/reforestree/mapping/final_dataset.csv",  # <-- Replace with your CSV filename
        n_folds=5,
        val_ratio=0.2,
        output_prefix="banana_splits"
    )