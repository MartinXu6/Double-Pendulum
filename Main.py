import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation

# Parameters
l1 = 3  # length of pendulum 1 (m)
l2 = 5  # length of pendulum 2 (m)
m1 = 2  # mass of pendulum 1 (kg)
m2 = 2  # mass of pendulum 2 (kg)
g = 9.81  # acceleration due to gravity (m/s^2)


initial_conditions = [
    [np.pi / 2, 0, np.pi, 0],
]

# Time array
t = np.linspace(0, 40, 1000)


def equations(y, t, l1, l2, m1, m2, g):
    theta1, omega1, theta2, omega2 = y

    delta_theta = theta2 - theta1

    den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(delta_theta) * np.cos(delta_theta)
    den2 = (l2 / l1) * den1

    dydt = [
        omega1,
        ((m2 * l1 * omega1 * omega1 * np.sin(delta_theta) * np.cos(delta_theta) +
          m2 * g * np.sin(theta2) * np.cos(delta_theta) +
          m2 * l2 * omega2 * omega2 * np.sin(delta_theta) -
          (m1 + m2) * g * np.sin(theta1)) / den1),
        omega2,
        ((-m2 * l2 * omega2 * omega2 * np.sin(delta_theta) * np.cos(delta_theta) +
          (m1 + m2) * g * np.sin(theta1) * np.cos(delta_theta) -
          (m1 + m2) * l1 * omega1 * omega1 * np.sin(delta_theta) -
          (m1 + m2) * g * np.sin(theta2)) / den2)
    ]
    return dydt


# Perform numerical integration for each set of initial conditions
results = []
for ic in initial_conditions:
    sol = odeint(equations, ic, t, args=(l1, l2, m1, m2, g))
    results.append(sol)

# Plot the results
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

for i, sol in enumerate(results):
    theta1 = sol[:, 0]
    theta2 = sol[:, 2]

    ax[0].plot(t, theta1, label=f'Set {i + 1}')
    ax[1].plot(t, theta2, label=f'Set {i + 1}')

ax[0].set_xlabel('Time [s]')
ax[0].set_ylabel('Theta1 [rad]')
ax[0].legend()
ax[0].set_title('Theta1 vs Time')

ax[1].set_xlabel('Time [s]')
ax[1].set_ylabel('Theta2 [rad]')
ax[1].legend()
ax[1].set_title('Theta2 vs Time')

plt.tight_layout()
plt.show()


def get_x1y1x2y2(the1, the2, L1, L2):
    return (L1 * np.sin(the1),
            -L1 * np.cos(the1),
            L1 * np.sin(the1) + L2 * np.sin(the2),
            -L1 * np.cos(the1) - L2 * np.cos(the2))


x1, y1, x2, y2 = get_x1y1x2y2(theta1, theta2, l1, l2)


def animate(i):
    ln1.set_data([0, x1[i], x2[i]], [0, y1[i], y2[i]])


fig, ax = plt.subplots(1, 1, figsize=(8, 8))
ax.set_facecolor('k')
ax.get_xaxis().set_ticks([])  # enable this to hide x axis ticks
ax.get_yaxis().set_ticks([])  # enable this to hide y axis ticks
ln1, = plt.plot([], [], 'go-', lw=3, markersize=8)
ax.set_ylim(-10, 10)
ax.set_xlim(-10, 10)
ani = animation.FuncAnimation(fig, animate, frames=1000, interval=50)
ani.save('pen.gif', writer='pillow', fps=25)
