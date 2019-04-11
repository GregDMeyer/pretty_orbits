
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

def plot_init(z):
    fig, ax = plt.subplots(figsize=(3,3))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    ax.axis('off')

    n = z.shape[0] // 4

    colors = ['C{}'.format(x) for x in range(n)]

    pts = plt.scatter(z[2*n::2, 0], z[2*n+1::2, 0], c=colors)
    lns = []
    for i in range(n):
        lns.append(plt.plot([], [], color=colors[i])[0])

    return fig, (pts, lns)

def plot_update(frame, draw_objs, z):
    n = z.shape[0] // 4
    pts, lns = draw_objs
    frame_data = np.vstack([z[2*n::2, frame], z[2*n+1::2, frame]]).T
    pts.set_offsets(frame_data)

    # add trailing lines
    trail_time = 10  # number of frames to trail for
    start = max(0, frame - trail_time)
    for i, ln in enumerate(lns):
        ln.set_data(z[2*(n+i), start:frame], z[2*(n+i)+1, start:frame])

    return [pts] + lns

def plot_soln(ts, zs):
    f,draw_objs = plot_init(zs)
    anim = FuncAnimation(f, lambda frame, d=draw_objs, z=zs: plot_update(frame, d, z), frames=np.arange(ts.size), blit=True, interval=25)
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
    ])

    tmin = 0
    tmax = 50
    tpts = tmax*10

    ts = np.linspace(tmin, tmax, tpts)

    r = solve_ivp(diff_eq, (tmin, tmax), z0, t_eval=ts, method='Radau')

    plot_soln(r.t, r.y)

if __name__ == "__main__":
    main()
