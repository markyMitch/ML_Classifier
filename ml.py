import pandas
import sklearn.model_selection
import sklearn.neural_network
import sklearn.metrics
import sklearn.preprocessing
import sklearn.metrics
import datetime
import numpy
import itertools
from typing import Union, Tuple, List

from google.colab import files

"""
To see the main script that runs the testing of different neural network configurations, head to
TestRig.master_test_script around line 258. This approach was designed to be modular and extendable, separating data
preparation, neural network training and output analysis sections of the work into individual classes. 

I wrote a simple docker file to run this in a CentOS container to enable easy config of environment pre-reqs.

Best config from testing was using the 'sgd' Stochastic Gradient Descent algorithm with hidden layers of sizes 
(9, 9, 2) and an average prediction accuracy of 0.7168
"""

class DataSet(object):  # data transfer object for data that has been split into test and training sets

    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test


class ClassifiedData(object):  # data transfer object for predictions and true values, used for results analysis

    def __init__(self, predictions: numpy.ndarray, true_values) -> None:
        self.predictions = predictions
        self.true_values = true_values


class NeuralNetConfig(object):  #  Tracks a hidden layers configuration and the best accuracy achieved with it

    def __init__(self, layers: Tuple[int], accuracy: float):
        self.layers = layers
        self.accuracy = accuracy


class DataParser(object):
    """
    Responsible for loading, cleaning, splitting and scaling data, so that the data can be handed of to a neural
    network ready for use.
    """

    def __init__(self):
        self.dataframe = None

    def load_data(self, filename: str) -> None:
        raise NotImplementedError

    def clean_data(self) -> None:
        raise NotImplementedError

    def split_data(self, y_column_name: str) -> DataSet:
        x = self.dataframe.drop(y_column_name, axis=1)
        y = self.dataframe[y_column_name]

        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y)
        return DataSet(x_train, x_test, y_train, y_test)

    def scale_data(self, dataset: DataSet) -> DataSet:
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(dataset.x_train)
        x_train_scaled = scaler.transform(dataset.x_train)
        x_test_scaled = scaler.transform(dataset.x_test)
        return DataSet(x_train_scaled, x_test_scaled, dataset.y_train, dataset.y_test)


class ExcelDataParser(DataParser):
    """
    Subclasses DataParser to implement specific methods for cleaning the Excel data provided. One of the main issues
    with the data is that the Excel doc has automatically comverted some of the ranges to date format (eg 10-14 becomes
    October 2014), so these are found and replaced within the imported dataframe
    """
    def __int__(self):
        super(ExcelDataParser, self).__init__()

    def load_data(self, filename: str) -> None:
        self.dataframe = pandas.read_excel(filename)

    def clean_data(self) -> None:
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
    """
    Represents a nerual network by wrapping the library class and controlling the config arguments provided to it
    eg layer sizes. This makes it easy to test different configurations by tweaking instance variables of this class.
    """

    def __init__(self):
        self.layer_sizes = None
        self.max_iter = None
        self.activation = None
        self.algorithm = None

    def classify_data(self, dataset: DataSet) -> ClassifiedData:
        if self.algorithm:
            algorithm = self.algorithm
        else:
            algorithm = "adam"
        mlp = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=self.layer_sizes, max_iter=self.max_iter,
                                                   solver=algorithm)
        mlp.fit(dataset.x_train, dataset.y_train)
        predictions = mlp.predict(dataset.x_test)
        return ClassifiedData(predictions, dataset.y_test)


class ClassifiedDataAnalyser(object):
    """
    This deals with taking ClassifiedData and finding the accuracy of the predictions, both for an individual instance
    and also over n interations to allow for repeated testing for greater accuracy. This class also handles writing
    test results to the log file so that they are not lost if a test set crashes mid-execution.
    """

    def __init__(self):
        self.iterations_analysed = 0
        self.running_average = 0
        self.log_file = None

    def analyse_data(self, classified_data: ClassifiedData) -> None:
        average = sklearn.metrics.accuracy_score(classified_data.true_values, classified_data.predictions)
        self.iterations_analysed = self.iterations_analysed + 1
        self.running_average = self.running_average + average

    def print_info(self, classified_data: ClassifiedData) -> None:
        print(sklearn.metrics.confusion_matrix(classified_data.true_values, classified_data.predictions))
        print(sklearn.metrics.classification_report(classified_data.true_values, classified_data.predictions))

    def get_average(self) -> float:
        return self.running_average / self.iterations_analysed

    def print_average(self) -> None:
        print(self.get_average())

    def reset_average(self) -> None:
        self.running_average = 0
        self.iterations_analysed = 0

    def set_log_file(self, filename: str) -> None:
        self.log_file = filename

    def log_config(self, message: str) -> None:
        print(message)
        try:
            with open(self.log_file, 'a') as log:
                log.write(f"{message}\n")
        except OSError as e:
            print(e)


class TestRig(object):
    """
    This class provides the top-level interface for conducting testing of different hidden layer sizes and neural
    network solving algorithms. The class splits this down into methods that handle different levels of abstraction,
    from running multiple iterations of the same test for reliability to testing different layer configurations and
    algorithms.
    """
    data_source = None
    STANDARD_ITERATIONS = 100
    DEFAULT_LOG = 'config_log.txt'

    def __init__(self, parser: ExcelDataParser, neural_net: NeuralNet, analyser: ClassifiedDataAnalyser):
        self.parser = parser
        self.neural_net = neural_net
        self.analyser = analyser
        self.analyser.set_log_file(self.DEFAULT_LOG)

    def set_data_source(self, data_source: str) -> None:
        parser.load_data(data_source)
        parser.clean_data()

    def set_hidden_layers(self, layers: Tuple[int]) -> None:
        self.neural_net.layer_sizes = layers

    def test_current_config(self, iterations: int) -> float:
        self.neural_net.max_iter = 20000

        for i in range(iterations):
            dataset = self.parser.split_data("Class")
            scaled_dataset = self.parser.scale_data(dataset)
            classified_data = self.neural_net.classify_data(scaled_dataset)
            self.analyser.analyse_data(classified_data)
        average_accuracy = self.analyser.get_average()
        self.analyser.reset_average()
        return average_accuracy

    def test_layer_sizes(self, num_layers: int, min: int, max: int) -> NeuralNetConfig:
        best_combination = None
        best_accuracy = 0.0

        layer_combinations = itertools.product(range(min, max), repeat=num_layers)
        for layer_config in layer_combinations:
            self.set_hidden_layers(layer_config)
            average_accuracy = self.test_current_config(self.STANDARD_ITERATIONS)
            self.analyser.log_config(f"{str(layer_config)} {str(average_accuracy)}")

            if average_accuracy > best_accuracy:
                best_combination = layer_config
                best_accuracy = average_accuracy

        self.analyser.log_config(f"BEST CONFIG FOR LAYER: {str(best_combination)} {str(best_accuracy)}")
        return NeuralNetConfig(best_combination, best_accuracy)

    def test_algorithm(self, algorithm: str) -> NeuralNetConfig:
        best_config = None

        self.neural_net.algorithm = algorithm
        self.analyser.set_log_file(self.DEFAULT_LOG)
        self.analyser.log_config(f"Starting testing on {algorithm}")
        for i in range(1,4):
            config = self.test_layer_sizes(i, 1, 10)
            if best_config is None or config.accuracy > best_config.accuracy:
                best_config = config

        self.analyser.log_config(f"BEST CONFIG FOR ALGORITHM: {best_config.layers} {best_config.accuracy}")
        return best_config

    def master_test_script(self) -> NeuralNetConfig:
        best_config = None
        best_algorithm = None
        algorithms = ["lbfgs", "sgd", "adam"]
        for algorithm in algorithms:
            config = self.test_algorithm(algorithm)
            if best_config is None or config.accuracy > best_config.accuracy:
                best_config = config
                best_algorithm = algorithm
        self.analyser.log_config(f"BEST CONFIG OVERALL: {best_algorithm} {best_config.layers} {best_config.accuracy}")
        return best_config

if __name__ == '__main__':
    # setup

    uploaded = files.upload()

    parser = ExcelDataParser()
    neural_net = NeuralNet()
    analyser = ClassifiedDataAnalyser()

    # run master test script on test rig for provided data
    test_rig = TestRig(parser, neural_net, analyser)
    test_rig.set_data_source("breast-cancer.xls")
    best_config = test_rig.master_test_script()
    print(best_config)
    files.download('config-log.txt')
