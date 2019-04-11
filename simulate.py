'''
A simulation of long-range interacting classical particles in 2D.

Uncomment the various lines in main() to see a few examples.

Copyright Greg Meyer (c) 2019
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from scipy.integrate import solve_ivp

def main():
    # plot_soln(three_body_eject)
    # plot_soln(two_body_elliptical)
    plot_soln(four_body)

    # save_soln(four_body, tmax=60, filename="fourbody.mp4")

# EXAMPLES

# two bodies start orbiting each other,
# a third interacts, ejecting one and replacing it
three_body_eject = np.array([
#   px  py
    0.5,  -0.05,
    -0.5, -0.05,
    0,    0.1,
#   qx  qy
    0, 1.08,
    0, 0.92,
    0,  -2,
])[np.newaxis].T

# simply two-body elliptical orbit
two_body_elliptical = np.array([
#   px  py
    -0.2,  0,
    0.2,   0,
#   qx  qy
    0, 0.5,
    0, -0.5,
])[np.newaxis].T

# four bodies
four_body = np.array([
#   px  py
    0.5,  -0.05,
    -0.5, -0.05,
    0,    0.05,
    0,    0.05,
#   qx  qy
    0, 2.08,
    0, 1.92,
    0.1,  -2,
    -0.1,  -2,
])[np.newaxis].T

def diff_eq(t, z):
    """
    Compute the first time derivative of z = (p,q) (momentum and position coordinates).
    Here, q = (x1, y1, x2, y2, ...)

    qdot = p
    pdot = F(q), the sum of forces from all other particles

    This function implements a central force between each pair of particles
    that scales as 1/r^alpha
    """

    n = len(z)//4

    p = z[:2*n]
    x = z[2*n::2]
    y = z[2*n+1::2]

    qdot = p

    pxdot = np.ndarray(p.size//2)
    pydot = np.ndarray(p.size//2)

    # pdot = F
    f_coeff = 0.1     # the coefficient on the force
    alpha = 2         # the exponent of the force decay
    smoothing = 0.05  # to make the potential not singular
    for i in range(n):

        # compute distance r to all particles
        r = np.sqrt((x - x[i])**2 + (y - y[i])**2 + smoothing**2)

        # compute all interactions, except for with itself
        pxdot[i] = 0
        pydot[i] = 0
        if i>0:
            pxdot[i] += -np.sum((x[i] - x[:i]) / r[:i]**(alpha+1))
            pydot[i] += -np.sum((y[i] - y[:i]) / r[:i]**(alpha+1))
        if i<n-1:
            pxdot[i] += -np.sum((x[i] - x[i+1:]) / r[i+1:]**(alpha+1))
            pydot[i] += -np.sum((y[i] - y[i+1:]) / r[i+1:]**(alpha+1))

    # scale our summed accelerations by the coefficient
    pxdot *= f_coeff
    pydot *= f_coeff

    # recombine our results into zdot
    pdot = np.vstack([pxdot, pydot]).T.flatten()
    zdot = np.hstack([pdot, qdot])

    return zdot


class ParticleAnimator:
    '''
    This class handles the creation and updating of the plot
    '''

    def __init__(self, z, tstep=0.1):
        '''
        z holds the initial conditions
        '''

        self.z = z
        self.tstep = tstep

        self.fig, ax = plt.subplots(figsize=(6,6))
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.set_position([0, 0, 1, 1])

        ax.axis('off')

        n = z.shape[0] // 4

        # make each particle a different color
        colors = ['C{}'.format(x) for x in range(n)]

        # put dots on each point
        self.pts = plt.scatter(z[2*n::2], z[2*n+1::2], c=colors, zorder=3)

        # add lines trailing off each point
        self.lns = []
        for i in range(n):
            self.lns.append(TrailingLine([], [], ax, 2.0, color=colors[i]))

    def update(self, frame):
        '''
        Compute the next time step, and update the plot accordingly
        '''

        trail_time = 20

        # update z for this time step
        r = solve_ivp(diff_eq, (0, self.tstep), self.z[:,-1], method='Radau')

        # keep at most trail_time previous locations.
        # if we already have that many, get rid of the oldest one
        # otherwise, add a new entry to the previous locations
        if self.z.shape[1] > trail_time:
            self.z[:,:-1] = self.z[:,1:]
        else:
            self.z = np.hstack([self.z, np.ndarray((self.z.shape[0],1))])

        # add the IVP result
        self.z[:,-1] = r.y[:,-1]

        n = self.z.shape[0] // 4

        # update the scatter points
        frame_data = np.vstack([self.z[2*n::2, -1], self.z[2*n+1::2, -1]]).T
        self.pts.set_offsets(frame_data)

        # update trailing lines
        for i, ln in enumerate(self.lns):
            ln.set_data(self.z[2*(n+i),:], self.z[2*(n+i)+1,:])

        return [self.pts] + [ln.lc for ln in self.lns]


class TrailingLine:
    '''
    This class plots a line that starts out with zero thickness and linearly
    increases to thickness max_width along its path.
    '''

    def __init__(self, x, y, ax, max_width, **kwargs):
        self.ax = ax
        self.max_width = max_width
        lw = self._compute_linewidths(len(x))

        self.lc = LineCollection(self._compute_segments(x,y), linewidths=lw, **kwargs)

        ax.add_collection(self.lc)

    def set_data(self, x, y):
        self.lc.set_segments(self._compute_segments(x, y))
        lw = self._compute_linewidths(len(x))
        self.lc.set_linewidth(lw)

    @classmethod
    def _compute_segments(cls, x, y):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments

    def _compute_linewidths(self, npts):
        return np.linspace(0, self.max_width, npts)[:-1]

def plot_soln(z0):
    part_anim = ParticleAnimator(z0)
    anim = FuncAnimation(part_anim.fig, part_anim.update, blit=True, interval=30)
    plt.show()

def save_soln(z0, tmax, filename, tstep=0.1):
    part_anim = ParticleAnimator(z0, tstep)
    nframes = int(tmax/tstep)
    anim = FuncAnimation(part_anim.fig, part_anim.update, blit=True, frames=np.arange(nframes), interval=30)

    print('Saving to file "{}"'.format(filename))
    anim.save(filename)

if __name__ == "__main__":
    main()
