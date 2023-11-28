import sys

from EDA_simple import EDA_simple_main
from EDA import EDA_main
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')


def prepare_data(df):
    x = df.drop('Final_Grade', axis=1)
    y = df['Final_Grade']

    numerical_features = x.select_dtypes(include=['float64', 'int64']).columns
    categorical_features = x.select_dtypes(include=['object']).columns
    categorical_features = categorical_features.delete(categorical_features.size - 1)
    multi_label_feature = ['Learning_Disabilities']

    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()
    multi_labeled_transformer = OneHotEncoder(sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('multi', multi_labeled_transformer, multi_label_feature)
        ]
    )

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

    return preprocessor, x_train, x_test, y_train, y_test


def model_training(preprocessor, x_train, y_train, regressor):
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'MSE: {mse}')


def regression_main(eda):
    if eda == "med":
        df = EDA_simple_main()
    elif eda == "reg":
        df = EDA_main()
    else:
        sys.exit("input not recognized")
    preprocessor, x_train, x_test, y_train, y_test = prepare_data(df)
    regressor = GradientBoostingRegressor()
    model = model_training(preprocessor, x_train, y_train, regressor)
    evaluate_model(model, x_test, y_test)


if __name__ == '__main__':
    eda = input('Type "med" for handling missing numerical data with medians, or type "reg" for LinearRegression\n')
    regression_main(eda)
