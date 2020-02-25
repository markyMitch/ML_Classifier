import pandas
import sklearn.model_selection
import sklearn.neural_network
import sklearn.metrics
import sklearn.preprocessing
import sklearn.metrics
import datetime
import numpy
from typing import Union, Tuple, List

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

class NeuralNetConfiguration(object):

    def __init__(self, average: float, layers: Tuple[int]) -> None:
        self.average = average
        self.layers = layers

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

    def __init__(self):
        self.iterations_analysed = 0
        self.running_average = 0
        self.best_config = None

    def analyse_data(self, classified_data: ClassifiedData):
        average = sklearn.metrics.accuracy_score(classified_data.true_values, classified_data.predictions)
        self.iterations_analysed = self.iterations_analysed + 1
        self.running_average = self.running_average + average

    def print_info(self, classified_data):
        print(sklearn.metrics.confusion_matrix(classified_data.true_values, classified_data.predictions))
        print(sklearn.metrics.classification_report(classified_data.true_values, classified_data.predictions))

    def get_average(self):
        return self.running_average / self.iterations_analysed

    def print_average(self):
        print(self.get_average())

    def reset_average(self):
        self.running_average = 0
        self.iterations_analysed = 0

    def log_config(self, neural_net_layers: Tuple[int]):
        try:
            with open('/data/config_log.txt', 'a') as log:
                log.write(f"{neural_net_layers} {self.get_average()}")
        except OSError as e:
            print(e)

    def save_config(self, neural_net_layers: Tuple[int]):
        if self.best_config is None or self.get_average() > self.best_config.average:
            self.best_config = NeuralNetConfiguration(self.get_average(), neural_net_layers)

if __name__ == '__main__':
    # setup
    parser = ExcelDataParser()
    parser.load_data('/data/breast-cancer.xls')
    parser.clean_data()

    analyser = ClassifiedDataAnalyser()

    # TODO outer loop here with different neural net setups
    neural_net = NeuralNet()
    neural_net.max_iter = 100000

    for layer_size in range(1,100):
        neural_net.layer_sizes = (layer_size,)
        for i in range(100):  # run a bunch of times with randomised test/train partition to get accurate average
            dataset = parser.split_data("Class")
            classified_data = neural_net.classify_data(dataset)
            analyser.analyse_data(classified_data)
        print(neural_net.layer_sizes)
        analyser.print_average()
        analyser.log_config((layer_size,))
        analyser.save_config((layer_size,))
        analyser.reset_average()

    for layer_one_size in range(1,100):
        for layer_two_size in range(1,100):
            neural_net.layer_sizes = (layer_one_size, layer_two_size)
            for i in range(100):
                dataset = parser.split_data("Class")
                classified_data = neural_net.classify_data(dataset)
                analyser.analyse_data(classified_data)
            print(neural_net.layer_sizes)
            analyser.print_average()
            analyser.log_config((layer_one_size, layer_two_size))
            analyser.save_config((layer_one_size, layer_two_size))
            analyser.reset_average()



    for i in range(100):  # run a bunch of times with randomised test/train partition to get accurate average
        dataset = parser.split_data("Class")
        classified_data = neural_net.classify_data(dataset)
        analyser.analyse_data(classified_data)




