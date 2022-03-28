import pandas as pd
from GMM import *
import plot

def IrisClassification(num_features, k=2):
    data = pd.read_csv('../Datasets/Iris.csv', sep=',', header=0)
    data = data.reset_index()
    data = data.sample(frac=1)

    train = data.iloc[:130, :]
    test = data.iloc[130:, :]
    

    if num_features == 1:
        cols = ['PetalLengthCm', 'PetalWidthCm', 'SepalLengthCm', 'SepalWidthCm']
        for col in cols:
            x = train[[col]]
            x = np.array(x)
            gmm = GMM(x, k)
            gmm.fit()
            y_pred = gmm.predict(test, [col])
            print(y_pred)
            print(test.iloc[:, -1].to_numpy())
            plot.plot_1D(gmm, x, col, [])

    else:
        replace_map = {'Species': {'Iris-virginica': 1, 'Iris-versicolor': 2,'Iris-setosa':3}}
        data.replace(replace_map, inplace=True)
        label=data[['Species']]
        cols=[['SepalLengthCm', 'PetalLengthCm'], ['SepalLengthCm', 'PetalWidthCm'], ['SepalLengthCm', 'SepalWidthCm'], ['PetalLengthCm', 'SepalWidthCm'], ['PetalWidthCm','PetalLengthCm'], ['SepalWidthCm','PetalLengthCm']]
        for col in cols:
            x = data[col]
            x = np.array(x)
            gmm = GMM(x, k)
            gmm.fit()
            plot.plot_2D(gmm, x, col, label)

def glassClassification(k=2):
    data = pd.read_csv('../Datasets/Glass.csv', sep=',', header=0)
    cols = ["RI","Na","Mg","Al","Si","K","Ca"]
    for col in cols:
        x = data[[col]]
        x = np.array(x)

        gmm = GMM(x, k)
        gmm.fit()
        plot.plot_1D(gmm, x, col, [])

def main():
    # glassClassification()
    IrisClassification(1, 3)

if __name__ == "__main__":
    main()
