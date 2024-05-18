# import libraries
from sympy import *
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# constants
G = Symbol("G")
M1 = Symbol("M1")
M2 = Symbol("M2")
L1 = Symbol("L1")
L2 = Symbol("L2")

# moment of inertia

I = 1 / 12 * M1 * L1 ** 2

# the angles, which are functions of time
T = symbols("T")
the1 = symbols("the1", cls=Function)
the2 = symbols("the2", cls=Function)
the1 = the1(T)
the2 = the2(T)

# working out centers of mass of two rods

x1 = L1 * sin(the1)
y1 = -L1 * cos(the1)
x2 = x1 + L2 * sin(the2)
y2 = y1 - L2 * cos(the2)

# calculating differentials
the1d = diff(the1, T)
the2d = diff(the2, T)
the1dd = diff(the1d, T)
the2dd = diff(the2d, T)
Vx1 = diff(x1, T)
Vx2 = diff(x2, T)
Vy1 = diff(y1, T)
Vy2 = diff(y2, T)

# calculation of the Lagrangian
# kinetic energy
KE1 = 1 / 2 * M1 * (Vx1 ** 2 + Vy1 ** 2)
KE2 = 1 / 2 * M2 * (Vx2 ** 2 + Vy2 ** 2)

# rotational kinetic energy about center of mass of each rod
RE = 1 / 2 * I * (the1d ** 2 + the2d ** 2)

# GPE
GPE = M1 * G * y1 + M2 * G * y2

# Lagrangian

Lag = (KE1 + KE2 + RE - GPE)

LE1 = diff(Lag, the1) - diff(diff(Lag, the1d), T).simplify()
LE2 = diff(Lag, the2) - diff(diff(Lag, the2d), T).simplify()

sols = solve([LE1, LE2], (the1dd, the2dd),
             simplify=False, rational=False)
dz1dt_f = lambdify((T, G, M1, M2, L1, L2, the1, the2, the1d, the2d), sols[the1dd])
dz2dt_f = lambdify((T, G, M1, M2, L1, L2, the1, the2, the1d, the2d), sols[the2dd])
dthe1dt_f = lambdify(the1d, the1d)
dthe2dt_f = lambdify(the2d, the2d)


def dSdt(S, T, G, M1, M2, L1, L2):
    the1, z1, the2, z2 = S
    return [
        dthe1dt_f(z1),
        dz1dt_f(T, G, M1, M2, L1, L2, the1, the2, z1, z2),
        dthe2dt_f(z2),
        dz2dt_f(T, G, M1, M2, L1, L2, the1, the2, z1, z2),
    ]


T = np.linspace(0, 40, 1001)
G = 9.81
M1 = 2
M2 = 1
L1 = 2
L2 = 1
ans = odeint(dSdt, y0=[1, -3, -1, 5], t=T, args=(G, M1, M2, L1, L2))

the1 = ans.T[0]
the2 = ans.T[2]
plt.plot(T, the2)
plt.show()


def get_x1y1x2y2(t, the1, the2, L1, L2):
    return (L1 * np.sin(the1),
            -L1 * np.cos(the1),
            L1 * np.sin(the1) + L2 * np.sin(the2),
            -L1 * np.cos(the1) - L2 * np.cos(the2))


x1, y1, x2, y2 = get_x1y1x2y2(T, ans.T[0], ans.T[2], L1, L2)


def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_facecolor('k')
ax.get_xaxis().set_ticks([])  # enable this to hide x axis ticks
ax.get_yaxis().set_ticks([])  # enable this to hide y axis ticks
ln1, = plt.plot([], [], 'g-', lw=3, markersize=8)
ax.set_ylim(-4, 4)
ax.set_xlim(-4, 4)
ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50)
ani.save('pen.gif', writer='pillow', fps=25)