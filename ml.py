import pandas
import sklearn.model_selection
import sklearn.neural_network
import sklearn.metrics
import sklearn.preprocessing
import sklearn.metrics
import datetime
import numpy
from typing import Union, Tuple

class DataSet(object):  # data transfer object

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

class ClassifiedData(object):  # another data transfer object

    def __init__(self, predictions: numpy.ndarray, true_values) -> None:
        self.predictions = predictions
        self.true_values = true_values

class DataParser(object):

    def __init__(self):
        self.dataframe = None

    def load_data(self, filename: str):
        raise NotImplementedError

    def clean_data(self):
        raise NotImplementedError

    def split_data(self, y_column_name: str) -> DataSet:
        x = self.dataframe.drop(y_column_name, axis=1)
        y = self.dataframe[y_column_name]

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y)
        return DataSet(x_train, x_test, y_train, y_test)


class ExcelDataParser(DataParser):

    def __int__(self):
        super(ExcelDataParser, self).__init__()

    def load_data(self, filename: str) -> None:
        self.dataframe = pandas.read_excel(filename)

    def clean_data(self):
        columns_with_ranged_data = ['age', 'tumor-size', 'inv-nodes']
        for column in columns_with_ranged_data:
            self.clean_range_data(column)

        columns_to_encode = ["menopause", "node-caps", "breast", "breast-quad", "irradiat"]
        self.dataframe = pandas.get_dummies(self.dataframe, columns=columns_to_encode, prefix=columns_to_encode)
        # encode the labels we are trying to classify
        label_encoder = sklearn.preprocessing.LabelEncoder()
        self.dataframe['Class'] = label_encoder.fit_transform(self.dataframe['Class'])

    def clean_range_data(self, column_name: str) -> None:

        self.dataframe[column_name] = self.dataframe[column_name].apply(self.clean_column_item_formatting)

        self.dataframe[column_name] = self.dataframe[column_name].apply(self.replace_range_with_num)

    def clean_column_item_formatting(self, value: Union[str, datetime.datetime]) -> str:
        if type(value) is datetime.datetime:
            return self.correct_format_error(value)
        else:
            return value

    def correct_format_error(self, value: datetime.datetime) -> str:
        d = value.day
        m = value.month
        y = value.year
        if d == 1:
            return f"{m}-{self.shorten_year(y)}"
        else:
            return f"{d}-{m}"

    def shorten_year(self, year: int) -> int:
        string_year = str(year)
        if len(string_year) == 4:
            if string_year[0:2] == "20":
                return int(string_year[2:])
        return year

    def replace_range_with_num(self, range_string: str) -> float:
        range_parts = range_string.split('-')
        difference = int(range_parts[1]) - int(range_parts[0])
        return int(range_parts[0]) + (difference / 2)


class NeuralNet(object):

    def __init__(self):
        self.layer_sizes = None
        self.max_iter = None
        self.activation = None
        self.algorithm = None

    def classify_data(self, dataset: DataSet) -> ClassifiedData:
        mlp = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=self.layer_sizes, max_iter=self.max_iter)
        mlp.fit(dataset.x_train, dataset.y_train)
        predictions = mlp.predict(dataset.x_test)
        return ClassifiedData(predictions, dataset.y_test)


class ClassifiedDataAnalyser(object):

    def __init__(self, classified_data: ClassifiedData) -> None:
        self.classified_data = classified_data

    def print_info(self):
        print(sklearn.metrics.confusion_matrix(self.classified_data.true_values, self.classified_data.predictions))
        print(sklearn.metrics.classification_report(self.classified_data.true_values, self.classified_data.predictions))


if __name__ == '__main__':
    # setup
    parser = ExcelDataParser()
    parser.load_data('/data/breast-cancer.xls')
    parser.clean_data()
    dataset = parser.split_data("Class")

    neural_net = NeuralNet()
    neural_net.layer_sizes = (20,20,20, 20, 20)
    neural_net.max_iter = 100000
    classified_data = neural_net.classify_data(dataset)

    analyser = ClassifiedDataAnalyser(classified_data)
    analyser.print_info()


