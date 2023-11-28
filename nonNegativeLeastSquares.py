import sys

import numpy as np
from scipy.optimize import minimize
from EDA_simple import EDA_simple_main
from EDA import EDA_main
from regressionModel import prepare_data, evaluate_model, model_training
import warnings

warnings.filterwarnings('ignore')


# Non-Negative Least Squares min || y - bx|| ** 2 s.t b >= 0
class NNLS:
    def fit(self, x, y):
        # objective function
        def objective_function(beta, x, y):
            return np.sum((x.dot(beta) - y) ** 2)

        constraints = ({'type': 'ineq', 'fun': lambda beta: beta})

        init_beta = np.zeros(x.shape[1])

        result = minimize(objective_function, init_beta, args=(x, y), constraints=constraints)

        # optimal coefficients
        self.coefficients = result.x

        return self

    def predict(self, x):
        return x.dot(self.coefficients)


def main(eda):
    if eda == "med":
        df = EDA_simple_main()
    elif eda == "reg":
        df = EDA_main()
    else:
        sys.exit("input not recognized")
    preprocessor, x_train, x_test, y_train, y_test = prepare_data(df)
    regressor = NNLS()
    model = model_training(preprocessor, x_train, y_train, regressor)
    evaluate_model(model, x_test, y_test)


if __name__ == '__main__':
    eda = input('Type "med" for handling missing numerical data with medians, or type "reg" for LinearRegression\n')
    main(eda)
