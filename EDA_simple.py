import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    plt.title(title)
    plt.show()


def categorical_plot(df, xaxis, yaxis, hue, kind, title):
    sns.catplot(data=df, x=xaxis, y=yaxis, hue=hue, kind=kind)
    plt.title(title)
    plt.show()


def bar_plot(df, xaxis, yaxis, title):
    sns.barplot(data=df, x=xaxis, y=yaxis)
    plt.title(title)
    plt.show()


def fill_missing_numerical_with_median(df, variable_name):
    df[variable_name] = df[variable_name].fillna(df[variable_name].median())


def fill_missing_categorical_with_unkown(df, variable_name):
    df[variable_name] = df[variable_name].fillna('Unkown')


def EDA_simple_main():
    path = 'data/MLhomeassignmentdata.xlsx'
    df = load_data(path)
    explore_structure(df)
    plots_manager(df)
    missing_numerical_manager(df)
    missing_categorical_manager(df)
    print('NaNs counts in data after fix')
    print(df.isna().sum())
    return df


def plots_manager(df):
    plot_histogram(df, 'Age', 'Students Age')
    categorical_plot(df, 'Family_Size', 'Age', 'Gender',
                     'violin', 'Family Size vs. Age of students')
    bar_plot(df, 'School_Absences', 'Final_Grade',
             "Student's School Absences vs. Final Grade")
    bar_plot(df, 'Mother_Education', 'Final_Grade',
             "Mother Education vs Student's Grade")
    bar_plot(df, 'Father_Education', 'Final_Grade',
             "Father Education vs. Student's Final Grade")


def missing_categorical_manager(df):
    fill_missing_categorical_with_unkown(df, 'Extra_Educational_Support')
    fill_missing_categorical_with_unkown(df, 'Family_Educational_Support')
    fill_missing_categorical_with_unkown(df, 'Extra_Paid_Classes')
    fill_missing_categorical_with_unkown(df, 'Extra_Curricular_Activities')
    fill_missing_categorical_with_unkown(df, 'Attended_Nursery_School')
    fill_missing_categorical_with_unkown(df, 'Wants_Higher_Education')
    fill_missing_categorical_with_unkown(df, 'Internet_Access_At_Home')
    fill_missing_categorical_with_unkown(df, 'In_Romantic_Relationship')
    fill_missing_categorical_with_unkown(df, 'Learning_Disabilities')


def missing_numerical_manager(df):
    fill_missing_numerical_with_median(df, 'Father_Education')
    fill_missing_numerical_with_median(df, 'Home_to_School_Travel_Time')
    fill_missing_numerical_with_median(df, 'Weekly_Study_Time')
    fill_missing_numerical_with_median(df, 'Family_Relationship_Quality')
    fill_missing_numerical_with_median(df, 'Free_Time_After_School')
    fill_missing_numerical_with_median(df, 'Going_Out_With_Friends')
    fill_missing_numerical_with_median(df, 'Workday_Alcohol_Consumption')
    fill_missing_numerical_with_median(df, 'Weekend_Alcohol_Consumption')
    fill_missing_numerical_with_median(df, 'Current_Health_Status')


if __name__ == '__main__':
    EDA_simple_main()
