import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st


def show_null_values(df: pd.DataFrame):
    """
    Show the number of null values.
    """
    plt.figure(figsize=(12, 8))
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    missing.sort_values(ascending=False, inplace=True)
    print(missing)
    missing.plot.bar()
    plt.show()


def show_distributions(df: pd.DataFrame, column: str):
    """
    Show the distribution of numerical features.
    """
    if df[column].dtype not in ['float', 'int']:
        raise Exception("Can only display distribution of numerical features")
    f = df[column]
    plt.figure(1)
    plt.title('Johnson SU')
    sns.distplot(f, kde=False, fit=st.johnsonsu)
    plt.figure(2)
    plt.title('Normal')
    sns.distplot(f, kde=False, fit=st.norm)
    plt.figure(3)
    plt.title('Log Normal')
    sns.distplot(f, kde=False, fit=st.lognorm)
    plt.show()


def show_all_num_distributions(df: pd.DataFrame,
                               num_columns,
                               plot_func=sns.distplot):
    """
    Show distributions of all numerical features. 

    Parameters
    ----------
    df: data; type: DataFrame.
    num_columns: numerical culumn names; type: Tuple | List | ndarray | None.
    plot_func: function to draw; ex: sns.distplot、sns.barplot or customized function.
    """
    f = pd.melt(df, value_vars=num_columns)
    g = sns.FacetGrid(f,
                      col="variable",
                      col_wrap=2,
                      sharex=False,
                      sharey=False)
    g = g.map(plot_func, "value")
    plt.show()


def show_pairs_relationship(df: pd.DataFrame, num_columns: list):
    """
    Show relationship between numerical column pairs.
    Parameters
    ----------
    df: data; type: DataFrame.
    num_columns: numerical culumn names; type: List.
    """
    sns.set()
    sns.pairplot(df[num_columns], size=2, kind="scatter", diag_kind="kde")
    plt.show()


def show_correlation(df: pd.DataFrame, num_columns):
    """
    Show correlation of numerical columns.
    Parameters
    ----------
    df: data; type: DataFrame.
    num_columns: numerical culumn names; type: List.
    """
    correlation = df[num_columns].corr()
    plt.figure(figsize=(8, 8))
    plt.title('Correlation of Numeric Features with Price', size=16)
    sns.heatmap(correlation, square=True)
    plt.show()


if __name__ == '__main__':
    data_path = "D:/download/data/house/train.csv"
    data = pd.read_csv(data_path)
    num_columns = data.select_dtypes(include=['float', 'int']).columns
    print(num_columns)
    # show_distributions(data, 'SalePrice')
    show_correlation(data, num_columns)
