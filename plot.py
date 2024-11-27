import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import cm

def main():
    df = pd.read_csv('speedups_results.csv')
    df = df.to_numpy()

    X = [2,4,8,16,24]
    Y = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
    Y, X = np.meshgrid(Y, X)
    Z = df[np.where(df[:, 2] == 8), 6]
    Z = np.reshape(Z, (5, 7))

    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #ax.plot_surface(X, Y, Z, cmap=cm.Blues)

    #ax.set(xticklabels=[], yticklabels=[], zticklabels=[])

    #fig.savefig('temp.pdf', dpi=fig.dpi)
    #ax.set_xlabel('threads')
    #ax.set_ylabel('num of points')
    #ax.set_zlabel('par_single speedup')
    #plt.show()

    X = [2,4,8,16,24]
    a = df[np.where(df[:, 2] == 8), :][0]
    a = a[np.where(a[:, 3] == 1000000), :]
    
    Y = a[0, :, 6]
    plt.plot(X, Y, label="par_single")

    Y = a[0, :, 5]
    plt.plot(X, Y, label="par_simd")

    Y = a[0, :, 4]
    plt.plot(X, Y, label="par")

    plt.legend()
    plt.show()

if __name__ == '__main__': 
    main()