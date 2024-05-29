import pandas as pd
from linearRegression import LinearRegression



def main():
    df = pd.read_csv('./data.csv')
    X = df['km']. values
    Y = df['price']. values

    lreg = LinearRegression(lr = 0.01, n_iters=1000)
    lreg.fit(X, Y)
    lreg.save_parameters()
    lreg.plot(X, Y)


if __name__ == '__main__':
    main()

