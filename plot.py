import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from tqdm import trange


def get_data():
    df = pd.read_csv(
        'https://query.data.world/s/ydtlqistcr56h7xx36ltobnvtsazvt')
    # print(df.head())
    # print(df.describe())
    # print(df.shape)
    # print(df.dtypes)
    return df


def make_plots(df: pd.DataFrame):
    # histogram
    plt.hist(df['AveragePrice'])
    plt.show()

    # bar plot
    # plt.bar(list(set(df['type'].values)), list(
    #     df.groupby('type')['AveragePrice'].agg('sum').values))
    # plt.show()
    # plt.close()


def make_more_plots(df: pd.DataFrame):

    sns.barplot(list(set(df['type'].values)), list(
        df.groupby('type')['AveragePrice'].agg('sum').values))
    plt.show()
    plt.close()

    # sns.scatterplot(x='Total Volume', y='AveragePrice', hue='type', data=df)
    # plt.show()
    # plt.close()

    # sns.pairplot(df.iloc[:, 8:11], palette="husl", height=5.5)
    # plt.show()
    # plt.close()

    # g = sns.catplot('AveragePrice', 'region', data=df,
    #                 hue='year',
    #                 palette='Blues',
    #                 kind='point',
    #                 join=False
    #                 )
    # plt.show()
    # plt.close()

    # rs = np.random.RandomState(11)
    # x = rs.gamma(2, size=1000)
    # y = -.5 * x + rs.normal(size=1000)
    # sns.jointplot(x, y, kind="hex", color="#4CB391")
    # plt.show()
    # plt.close()


def make_basic_plots():

    # line plot
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    plt.scatter(x, y, label='scatter plot')
    plt.legend()
    plt.xlabel('some x')
    plt.ylabel('some y')
    plt.show()
    plt.close()

    # scatter plot
    x = np.linspace(0, 1000, 100)
    err = 100 * np.random.randn(100)
    y1 = 9.0 * x + 2.1 + err
    y2 = 7.2 * x + 9.8 + err
    plt.plot(x, y1, label='y1')
    plt.plot(x, y2, label='y2')
    plt.xlabel('some x')
    plt.ylabel('some y')
    plt.title('x vs y')
    plt.legend()
    plt.show()
    plt.close()


def bonus(df: pd.DataFrame):
    months = {'01': 'Jan', '02': 'Feb', '03': 'Mar', '04': 'Apr', '05': 'May', '06': 'Jun',
              '07': 'Jul', '08': 'Aug', '09': 'Sep', '10': 'Oct', '11': 'Nov', '12': 'Dec'}


def tqq():
    sum = 0
    for _ in range(10):
        sum += 1
        time.sleep(0.5)
    print('sum is: {}'.format(sum))


if __name__ == '__main__':
    dataframe = get_data()

    # matplotlib
    # make_basic_plots()
    # make_plots(dataframe)

    # seaborn
    # make_more_plots(dataframe)

    # bonus(dataframe)
    # tqq()
    pass
