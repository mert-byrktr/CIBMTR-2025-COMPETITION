import pandas as pd
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, train_path, test_path):
        # Configure pandas display options
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.max_rows', 500)
        
        # Load data
        self.train = pd.read_csv(train_path)
        self.test = pd.read_csv(test_path)
        
        # Define column groups
        self.rmv = ["ID", "efs", "efs_time", "y"]
        self.features = [c for c in self.train.columns if c not in self.rmv]
        self.cats = []
        self.num_cols = []
        
    def print_data_info(self):
        print("Test shape:", self.test.shape)
        print("Train shape:", self.train.shape)
        print(self.train.head())
        
    def plot_efs_distribution(self):
        plt.hist(self.train.loc[self.train.efs==1, "efs_time"], bins=100, label="efs=1, Yes Event")
        plt.hist(self.train.loc[self.train.efs==0, "efs_time"], bins=100, label="efs=0, Maybe Event")
        plt.xlabel("Time of Observation, efs_time")
        plt.ylabel("Density")
        plt.title("Times of Observation. Either time to event, or time observed without event.")
        plt.legend()
        plt.show()
        
    def identify_feature_types(self):
        # Identify categorical features and handle missing values
        for c in self.features:
            if self.train[c].dtype == "object":
                self.cats.append(c)
                self.train[c] = self.train[c].fillna("NAN")
                self.test[c] = self.test[c].fillna("NAN")
        
        self.num_cols = [c for c in self.features if c not in self.cats]
        
        print(f"There are {len(self.features)} FEATURES: {self.features}")
        print(f"In these features, there are {len(self.cats)} CATEGORICAL FEATURES: {self.cats}")
        print(f"In these features, there are {len(self.num_cols)} NUMERICAL FEATURES: {self.num_cols}")
        
    def preprocess_features(self):
        # Combine datasets for preprocessing
        combined = pd.concat([self.train, self.test], axis=0, ignore_index=True)
        
        print("We LABEL ENCODE the CATEGORICAL FEATURES: ", end="")
        for c in self.features:
            # Label encode categorical features
            if c in self.cats:
                print(f"{c}, ", end="")
                combined[c], _ = combined[c].factorize()
                combined[c] -= combined[c].min()
                combined[c] = combined[c].astype("int32")
                combined[c] = combined[c].astype("category")
            # Reduce precision of numerical features
            else:
                if combined[c].dtype == "float64":
                    combined[c] = combined[c].astype("float32")
                if combined[c].dtype == "int64":
                    combined[c] = combined[c].astype("int32")
        
        # Split back into train and test
        self.train = combined.iloc[:len(self.train)].copy()
        self.test = combined.iloc[len(self.train):].reset_index(drop=True).copy()
        
    def process(self):
        """Run the complete preprocessing pipeline"""
        self.print_data_info()
        # self.plot_efs_distribution()
        self.identify_feature_types()
        self.preprocess_features()
        return self.train, self.test

# Example usage:
if __name__ == "__main__":
    train_path = r"C:\Projects\CIBMTR\data\cibmtr\train.csv"
    test_path = r"C:\Projects\CIBMTR\data\cibmtr\test.csv"
    
    preprocessor = DataPreprocessor(train_path, test_path)
    train_processed, test_processed = preprocessor.process()