#**************************** 6x6 Matrix ****************************
import numpy as np

data = np.random.randint(1, 100, size=(6, 6))
print(f"Original 6x6 Matrix:\n{data}")

#**************************** Calculate Mean and Median using Pandas ****************************
import pandas as pd

def calculate_mean_and_median(data):
    series = pd.Series(data)
    mean = series.mean()
    median = series.median()
    return mean, median

data = [1, 2, 3, 4, 5]
mean, median = calculate_mean_and_median(data)
print(f"\nMean: {mean}, Median: {median}")

#**************************** Show Dataset Shape Information ****************************
import sklearn.datasets as datasets

def datasetShapeInfo(dataset_name):
    available_datasets = {
        'iris': datasets.load_iris,
        'wine': datasets.load_wine,
        'breast_cancer': datasets.load_breast_cancer
    }
    if dataset_name not in available_datasets:
        raise ValueError(f"Dataset '{dataset_name}' is not available. Choose from {list(available_datasets.keys())}.")
   
    dataset = available_datasets[dataset_name]()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    return df.shape

dataset_name = "iris"
print(f"\nShape of {dataset_name} dataset:", datasetShapeInfo(dataset_name))

#**************************** Load Dataset and Display First 7 Rows ****************************
def load_dataset(dataset_name):
    available_datasets = {
        'iris': datasets.load_iris,
        'wine': datasets.load_wine,
        'breast_cancer': datasets.load_breast_cancer
    }
    if dataset_name not in available_datasets:
        raise ValueError(f"Dataset '{dataset_name}' is not available. Choose from {list(available_datasets.keys())}.")
   
    dataset = available_datasets[dataset_name]()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    return df

df = load_dataset("iris")
print("\nFirst 7 Rows of Iris Dataset:")
print(df.iloc[:7, :])

#**************************** Mean Sepal Length for Each Iris Species ****************************
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame
grouped = df.groupby("target")["sepal length (cm)"].mean()

print("\nMean Sepal Length for Each Species:")
print(grouped)

#**************************** Correlation Heatmap of Iris Dataset ****************************
import seaborn as sns
import matplotlib.pyplot as plt

corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap of Iris Dataset")
plt.show()

#**************************** Draw Histograms for Datasets ****************************
from sklearn.datasets import load_wine, load_breast_cancer

def drawHistogram(dataset_name):
    available_datasets = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer
    }
    if dataset_name not in available_datasets:
        raise ValueError(f"Dataset '{dataset_name}' is not available.")
   
    dataset = available_datasets[dataset_name]()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    
    df.hist(figsize=(10, 8))
    plt.suptitle(f'Histograms for {dataset_name} dataset', fontsize=16)
    plt.tight_layout()
    plt.show()

drawHistogram("iris")

#**************************** Simple Linear Regression on Iris Dataset ****************************
from sklearn.linear_model import LinearRegression

X = df[["sepal length (cm)"]]
y = df["petal length (cm)"]

model = LinearRegression()
model.fit(X, y)

print("\nLinear Regression Results:")
print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

#**************************** Describe User Input Numbers using Pandas ****************************
import numpy as np

data = np.array([10, 20, 30, 40, 50])
sf = pd.Series(data)
print("\nStatistical Summary of Sample Data:")
print(sf.describe())

#**************************** Min-Max Scaling of Sepal Length ****************************
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[["sepal length (cm)"]])

print("\nFirst 10 Min-Max Scaled Values for Sepal Length:")
print(scaled[:10].flatten())

#**************************** Display Summary of CSV File ****************************
def displaySummary(df):
   print(f"\nCSV Summary:\n{df.info()}\n\n{df.describe()}")

# Uncomment below if you have a file named 'store.csv'
# data = pd.read_csv('store.csv')
# displaySummary(data)

#**************************** Scatterplot: Sepal Length vs Petal Length ****************************
plt.scatter(df["sepal length (cm)"], df["petal length (cm)"], alpha=0.7)
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.title("Scatterplot: Sepal Length vs Petal Length")
plt.show()

#**************************** Train-Test Split for Dataset ****************************
from sklearn.model_selection import train_test_split

def splitDataset(dataset_name, test_size, random_state):
    available_datasets = {
        'iris': datasets.load_iris,
        'wine': datasets.load_wine,
        'breast_cancer': datasets.load_breast_cancer
    }
    if dataset_name not in available_datasets:
        raise ValueError(f"Dataset '{dataset_name}' is not available.")
   
    dataset = available_datasets[dataset_name]()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    
    train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)
    return train_set, test_set

train_set, test_set = splitDataset("iris", 0.3, 42)
print(f"\nTraining Set Shape: {train_set.shape}")
print(f"Testing Set Shape: {test_set.shape}")

#**************************** Statistical Analysis of Datasets (Mean, Median, Variance, STD) ****************************
def statisticalAnalysis(dataset_name):
    available_datasets = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer
    }
    if dataset_name not in available_datasets:
        raise ValueError(f"Dataset '{dataset_name}' is not available.")
   
    dataset = available_datasets[dataset_name]()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    
    std = df.std()
    mean = df.mean()
    median = df.median()
    variance = df.var()
    
    print(
        f"\nStandard Deviation:\n{std}\n\n"
        f"Mean:\n{mean}\n\n"
        f"Median:\n{median}\n\n"
        f"Variance:\n{variance}"
        )

statisticalAnalysis("iris")
