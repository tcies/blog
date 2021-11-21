import IPython
import numpy as np
import scipy.optimize

import bad_opt


def cos_sin(alpha):
    return np.cos(alpha), np.sin(alpha)


def R_x(alpha):
    c, s = cos_sin(alpha)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def R_y(alpha):
    c, s = cos_sin(alpha)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])


def R_z(alpha):
    c, s = cos_sin(alpha)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])


def R_from_rpy(rpy):
    return R_x(rpy[0]) @ R_y(rpy[1]) @ R_z(rpy[2])


def main():
    p = bad_opt.Problem()
    R_opt = scipy.optimize.minimize(lambda x: p.error(R_from_rpy(x)), np.zeros(3))
    print('Roll, pitch and yaw are:')
    print(R_opt.x)
    print('Rotation matrix is:')
    R = R_from_rpy(R_opt.x)
    print(R)
    print('Basis vector norms:')
    print(np.linalg.norm(R, axis=0))
    print('Orthogonality checks:')
    for i in range(3):
        rolled = np.roll(R, i, axis=1)
        z_from_xy = np.cross(rolled[:, 0], rolled[:, 1])
        print(np.linalg.norm(z_from_xy - rolled[:, 2]))
    print('x unit vector should be around:')
    print(p.points_center / np.linalg.norm(p.points_center))


if __name__ == '__main__':
    main()
