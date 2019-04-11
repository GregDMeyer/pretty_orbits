
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

def diff_eq(t, z):
    """
    Interacting particles in 2D, with r^-2 interaction
    """

    n = len(z)//4

    zdot = np.ndarray(z.size)

    # qdot = p
    zdot[2*n:] = z[:2*n]

    # pdot = F
    f_coeff = 0.1
    alpha = 3
    for i in range(n):

        # compute all hypotenuses
        r = np.maximum(0.05, np.hypot(z[2*n::2]-z[2*n+2*i], z[2*n+1::2]-z[2*(n+i)+1]))

        zdot[2*i] = 0
        zdot[2*i+1] = 0
        if i>0:
            zdot[2*i] += -np.sum((z[2*(n+i)] - z[2*n:2*(n+i):2]) / r[:i]**alpha)
            zdot[2*i+1] += -np.sum((z[2*(n+i)+1] - z[2*n+1:2*(n+i)+1:2]) / r[:i]**alpha)
        if i<n-1:
            zdot[2*i] += -np.sum((z[2*(n+i)] - z[2*(n+i+1)::2]) / r[i+1:]**alpha)
            zdot[2*i+1] += -np.sum((z[2*(n+i)+1] - z[2*(n+i+1)+1::2]) / r[i+1:]**alpha)

    zdot[:2*n] *= f_coeff

    return zdot


class ParticleAnimator:

    def __init__(self, z):

        self.z = z

        self.fig, ax = plt.subplots(figsize=(3,3))
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')

        ax.axis('off')

        n = z.shape[0] // 4

        colors = ['C{}'.format(x) for x in range(n)]

        self.pts = plt.scatter(z[2*n::2], z[2*n+1::2], c=colors)
        self.lns = []
        for i in range(n):
            self.lns.append(plt.plot([], [], color=colors[i])[0])

    def update(self, frame):

        trail_time = 100
        tstep = 0.1

        r = solve_ivp(diff_eq, (0, tstep), self.z[:,-1], method='Radau')

        if self.z.shape[1] > trail_time:
            self.z[:,:-1] = self.z[:,1:]
        else:
            self.z = np.hstack([self.z, np.ndarray((self.z.shape[0],1))])

        self.z[:,-1] = r.y[:,-1]

        n = self.z.shape[0] // 4

        frame_data = np.vstack([self.z[2*n::2, -1], self.z[2*n+1::2, -1]]).T
        self.pts.set_offsets(frame_data)

        # add trailing lines
        for i, ln in enumerate(self.lns):
            ln.set_data(self.z[2*(n+i),:], self.z[2*(n+i)+1,:])

        return [self.pts] + self.lns

def plot_soln(zs):
    part_anim = ParticleAnimator(zs)
    anim = FuncAnimation(part_anim.fig, part_anim.update, blit=True, interval=20)
    plt.show()

def main():
    z0 = np.array([
    #   px  py
        0.5,  -0.05,
        -0.5, -0.05,
        0,    0.1,
    #   qx  qy
        0, 1.11,
        0, 0.89,
        0,  -2,
    ])[np.newaxis].T

    # z0 = np.array([
    # #   px  py
    #     -0.2,  0,
    #     0.2,   0,
    # #   qx  qy
    #     0, 0.5,
    #     0, -0.5,
    # ])[np.newaxis].T

    plot_soln(z0)

if __name__ == "__main__":
    main()
