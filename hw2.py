import numpy as np


class Solution:
    def read_data(self, fileName):
        name = fileName[fileName.rfind('/', fileName) + 1:]
        data = np.genfromtxt(fileName, delimiter=',', skip_header=1)
        if name == "X_train":
            self.x_train = data
        elif name == "Y_train":
            self.y_train = data.reshape(-1, 1)
        elif name == "X_test":
            self.x_test = data

    @staticmethod
    def normalize(x, isTrain, mean=None, std=None):
        if isTrain:
            std = np.std(x, axis=0)
            mean = np.mean(x, axis=0)
        x = (x - mean) / std
        return mean, std

    @staticmethod
    def train_validate_split(x, y, percent=0.15):
        length = np.floor(x.shape[0] * (1 - percent))
        x_train = x[0: length]
        y_train = y[0: length]
        x_dev = x[length:]
        y_dev = y[length:]
        return x_train, y_train, x_dev, y_dev

    @staticmethod
    def _sigmoid(z):
        val = np.clip(1 / (1.0 + np.exp(-z)), 1e-6, 1 - 1e-6)
        return val

    @staticmethod
    def _get_prob(x, w, b):
        return Solution._sigmoid(x.dot(w) + b)

    @staticmethod
    def _infer(x, w, b):
        return np.round(Solution._get_prob(x, w, b))

    @staticmethod
    def _cross_entropy(y_pred, y_label):
        cross_entropy = -y_label.dot(np.log(y_pred)) - \
            (1-y_label).dot(np.log(1-y_pred))
        return cross_entropy

    @staticmethod
    def _loss(y_pred, y_label, lamda, w):
        return Solution._cross_entropy(y_pred, y_label) + lamda * (w ** 2)

    @staticmethod
    def _accuracy(y_pred, y_label):
        return np.sum(y_pred == y_label) / len(y_pred)

    @staticmethod
    def train(x, y, regularization="True"):
        pass


def main():
    sol = Solution()
    x_train_filename = "D:/data/hw2/X_train"
    y_train_filename = "D:/data/hw2/Y_train"
    x_test_filename = "D:/data/hw2/X_test"

    sol.read_data(x_train_filename)
    sol.read_data(y_train_filename)
    sol.read_data(x_test_filename)

    mean, std = sol.normalize(sol.x_train, isTrain="True")
    x_train, y_train, x_dev, y_dev = sol.train_validate_split(
        sol.x_train, sol.y_train)
    w, b = sol.train(x_train, y_train)
    x_test = sol.x_test
    sol.normalize(x_test, False, mean, std)
    

if __name__ == "__main__":
    main()
