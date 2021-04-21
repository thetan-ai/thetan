import numpy as np
import numpy.testing as npt



if __name__ == '__main__':
    print('check')

    x = 10 * np.random.rand(3000) + 2
    res = np.histogram(x, bins=100, range=(0, x.max()))

    print(res)

    x2 = 10 * np.random.rand(3000) + 2
    res2 = np.histogram(x2, bins=100, range=(0, x.max()))

    x2

    pass
