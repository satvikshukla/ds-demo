import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def get_data():
    df = pd.read_csv(
        'https://query.data.world/s/ydtlqistcr56h7xx36ltobnvtsazvt')
    # print(df.head())
    # print(df.describe())
    # print(df.shape)
    # print(df.dtypes)
    return df


def make_plots(df: pd.DataFrame):
    # plt.hist(df['AveragePrice'])
    # plt.show()

    # sns.scatterplot(x='Total Volume', y='AveragePrice', hue='type', data=df)
    # sns.pairplot(df.iloc[:,8:11], palette="husl",height=5.5)
    g = sns.catplot('AveragePrice','region',data=df,
                   hue='year',
                   palette='Blues',
                   kind='point',
                   join=False
              )
    plt.show()

    rs = np.random.RandomState(11)

    x = rs.gamma(2, size=1000)
    y = -.5 * x + rs.normal(size=1000)

    sns.jointplot(x, y, kind="hex", color="#4CB391")
    plt.show()

def make_basic_plots():

    # line plot
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    plt.scatter(x, y, label='scatter plot')
    plt.legend()
    plt.xlabel('some x')
    plt.ylabel('some y')
    plt.show()

    # scatter plot
    x = np.linspace(0, 1000, 100)
    err = 100 * np.random.randn(100)
    y1 = 9.0 * x + 2.1 + err
    y2 = 7.2 * x + 9.8 + err
    plt.plot(x, y1, label='y1')
    plt.plot(x, y2, label='y2')
    plt.xlabel('some x')
    plt.ylabel('some y')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    dataframe = get_data()
    # make_basic_plots()
    make_plots(dataframe)
