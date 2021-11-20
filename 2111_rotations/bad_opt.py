import numpy as np
import scipy.optimize


class Problem(object):
    def __init__(self):
        self.points_center = np.array([2., 1., 0.5])
        self.points = np.random.normal(self.points_center, size=(30, 3))

    def error(self, R):
        R_x_ax = R[:, 0]
        dot_prods = [np.dot(p, R_x_ax) for p in self.points]
        return -np.sum(dot_prods)


def main():
    p = Problem()
    R_opt = scipy.optimize.minimize(lambda x: p.error(x.reshape((3, 3))), np.eye(3).reshape((-1)))
    print(R_opt.x.reshape((3, 3)))


if __name__ == '__main__':
    main()
