import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


def load_data(path):
    df = pd.read_excel(path)
    return df


def explore_structure(df):
    print('dataframe shape: {}'.format(df.shape))
    print('dataframe info:')
    print(df.info())
    print('dataframe describe: ')
    print(df.describe())
    print('NaNs count:')
    print(df.isna().sum())
    print('duplicate rows count')
    print(df.shape[0] - df.drop_duplicates().shape[0])


def plot_histogram(df, variable_name, title):
    sns.histplot(data=df, x=variable_name)
    sns.set(rc={'figure.figsize': (8, 8)})
    plt.title(title)
    plt.show()


def bar_plot(df, xaxis, yaxis, title):
    sns.barplot(data=df, x=xaxis, y=yaxis)
    sns.set(rc={'figure.figsize': (8, 8)})
    plt.title(title)
    plt.show()


def categorical_plot(df, xaxis, yaxis, hue, kind, title):
    sns.catplot(data=df, x=xaxis, y=yaxis, hue=hue, kind=kind)
    sns.set(rc={'figure.figsize': (8, 9)})
    plt.title(title)
    plt.show()


def check_nan_rows(df):
    mask = df.isna()
    mask = mask.any(axis=1)
    df_nas = df[mask]
    print("shape of dataframe that includes only rows with NaN is: ".format(df_nas.shape))
    print("rows with NaNs: \n")
    print(df_nas[['Father_Education', 'Home_to_School_Travel_Time', 'Weekly_Study_Time',
                  'Extra_Educational_Support', 'Family_Educational_Support', 'Extra_Paid_Classes',
                  'Extra_Curricular_Activities', 'Attended_Nursery_School', 'Wants_Higher_Education',
                  'Internet_Access_At_Home', 'Family_Relationship_Quality', 'Final_Grade']][:5])


def check_normality(df, variable_name, title, bins=10):
    plt.hist(x=df[variable_name], bins=bins)
    plt.xlabel(variable_name)
    plt.title(title)
    plt.show()


def split_into_train_test(df, variable_name):
    data = df.drop('Learning_Disabilities', axis=1)
    data = data.select_dtypes(['int64', 'float64'])
    test_data = data[data[variable_name].isnull()]
    data.dropna(inplace=True)
    x_train = data.drop(variable_name, axis=1)
    x_test = test_data.drop(variable_name, axis=1)
    y_train = data[variable_name]
    return x_train, y_train, x_test


def handle_nans_in_test_batch(x_test):
    for col_name in x_test:
        x_test[col_name] = x_test[col_name].fillna(x_test[col_name].median())
    return x_test


def predict_nans(df, col_name):
    x_train, y_train, x_test = split_into_train_test(df, col_name)
    x_test = handle_nans_in_test_batch(x_test)
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return y_pred


def check_categorical(df, col):
    print(df[col].describe())
    print(df[col].value_counts())
    print(df[col].isna().sum())


def fill_missing_categorical_unkown(df, col):
    df[col] = df[col].fillna('Unkown')


def EDA_main():
    path = 'data/MLhomeassignmentdata.xlsx'
    df = load_data(path)
    plots_manager(df)
    check_normality(df, 'Father_Education', "Normality Check", 5)
    predict_nans_manager(df)
    check_categorical(df, 'Learning_Disabilities')
    missing_categorical_manager(df)
    print('NaNs counts in data after fix')
    print(df.isna().sum())
    return df


def missing_categorical_manager(df):
    fill_missing_categorical_unkown(df, 'Extra_Educational_Support')
    fill_missing_categorical_unkown(df, 'Family_Educational_Support')
    fill_missing_categorical_unkown(df, 'Extra_Paid_Classes')
    fill_missing_categorical_unkown(df, 'Extra_Curricular_Activities')
    fill_missing_categorical_unkown(df, 'Attended_Nursery_School')
    fill_missing_categorical_unkown(df, 'Wants_Higher_Education')
    fill_missing_categorical_unkown(df, 'Internet_Access_At_Home')
    fill_missing_categorical_unkown(df, 'In_Romantic_Relationship')
    fill_missing_categorical_unkown(df, 'Learning_Disabilities')


def plots_manager(df):
    plot_histogram(df, 'Age', 'Students Age')
    bar_plot(df, 'School_Absences', 'Final_Grade',
             'School Absences vs. Students Final Grade')
    bar_plot(df, 'Mother_Education', 'Final_Grade',
             "Mother Education vs. Students' Final Grade")
    bar_plot(df, 'Father_Education', 'Final_Grade',
             "Father Education vs. Students' Final Grade")
    categorical_plot(df, 'Extra_Educational_Support', 'Final_Grade', 'Gender',
                     'box', "Extra Educationa Support vs. Final Grade")
    categorical_plot(df, 'Extra_Paid_Classes', 'Final_Grade', 'Gender',
                     'box', "Extra Paid Classes vs. Final Grade")


def predict_nans_manager(df):
    df.loc[np.isnan(df['Father_Education']), 'Father_Education'] = predict_nans(df, 'Father_Education')
    df.loc[np.isnan(df['Home_to_School_Travel_Time']), 'Home_to_School_Travel_Time'] = (
        predict_nans(df, 'Home_to_School_Travel_Time'))
    df.loc[np.isnan(df['Weekly_Study_Time']), 'Weekly_Study_Time'] = (
        predict_nans(df, 'Weekly_Study_Time'))
    df.loc[np.isnan(df['Family_Relationship_Quality']), 'Family_Relationship_Quality'] = \
        predict_nans(df, 'Family_Relationship_Quality')
    df.loc[np.isnan(df['Free_Time_After_School']), 'Free_Time_After_School'] = (
        predict_nans(df, 'Free_Time_After_School'))
    df.loc[np.isnan(df['Going_Out_With_Friends']), 'Going_Out_With_Friends'] = (
        predict_nans(df, 'Going_Out_With_Friends'))
    df.loc[np.isnan(df['Workday_Alcohol_Consumption']), 'Workday_Alcohol_Consumption'] = (
        predict_nans(df, 'Workday_Alcohol_Consumption'))
    df.loc[np.isnan(df['Weekend_Alcohol_Consumption']), 'Weekend_Alcohol_Consumption'] = (
        predict_nans(df, 'Weekend_Alcohol_Consumption'))
    df.loc[np.isnan(df['Current_Health_Status']), 'Current_Health_Status'] = (
        predict_nans(df, 'Current_Health_Status'))


if __name__ == '__main__':
    EDA_main()
