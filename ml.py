import pandas
import sklearn.model_selection
import sklearn.neural_network
import sklearn.metrics
import sklearn.preprocessing
import sklearn.metrics
import datetime
from typing import Union

data = pandas.read_excel('/data/breast-cancer.xls')
print('test')
print(data)

def prep_data(data):

    # TODO need to change range columns to use mean value and change category columns to use one-hot encoding
    print(type(data))
    clean_range_data(data, 'age')
    clean_range_data(data, 'tumor-size')
    clean_range_data(data, 'inv-nodes')
    data = pandas.get_dummies(data, columns=["menopause", "node-caps", "breast", "breast-quad", "irradiat"],
                       prefix=["menopause", "node-caps", "breast", "breast-quad", "irradiat"])
    print(data.head())

    # encode the labels we are trying to classify
    label_encoder = sklearn.preprocessing.LabelEncoder()
    data['Class'] = label_encoder.fit_transform(data['Class'])

    X = data.drop('Class', axis=1)
    y = data['Class']

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)

    mlp = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(20,20,20),max_iter=500)

    mlp.fit(X_train, y_train)

    predictions = mlp.predict(X_test)
    print(predictions)

    # check accuracy
    print(sklearn.metrics.confusion_matrix(y_test, predictions))
    print(sklearn.metrics.classification_report(y_test, predictions))

def clean_range_data(data: pandas.core.frame.DataFrame, column_name: str) -> None:
    labels = {}

    # run initial loop to correct any data in date format

    data[column_name] = data[column_name].apply(clean_column_item_formatting)

    for value in data[column_name]:
        if value not in labels:
            labels[value] = replace_range_with_num(value)
    replacement_data = {column_name: labels}
    data.replace(replacement_data, inplace=True)

def clean_column_item_formatting(value: Union[int, datetime.datetime]) -> str:
    if type(value) is datetime.datetime:
        return correct_format_error(value)
    else:
        return value

def correct_format_error(value: datetime.datetime) -> str:
    d = value.day
    m = value.month
    y = value.year
    if d == 1:
        return f"{m}-{shorten_year(y)}"
    else:
        return f"{d}-{m}"

def shorten_year(year: int) -> int:
    string_year = str(year)
    if len(string_year) == 4:
        if string_year[0:2] == "20":
            return int(string_year[2:])
    return year

def replace_range_with_num(range_string: str) -> float:
    range_parts = range_string.split('-')
    difference = int(range_parts[1]) - int(range_parts[0])
    return int(range_parts[0]) + (difference / 2)

if __name__ == '__main__':
    prep_data(data)