import argparse
import mlflow
from sklearn import datasets
from sklearn import svm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", default=100, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    args = parser.parse_args()

    print(args.batch_size, args.epochs)

    clf = svm.SVC(gamma='scale')
    iris = datasets.load_iris()

    x, y = iris.data, iris.target
    clf.fit(x, y)
    print(x)
    mlflow.sklearn.log_model(clf, "svm_model")
