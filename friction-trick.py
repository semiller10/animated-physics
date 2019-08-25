from math import cos, sin
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.constants
import sys
import time

g = scipy.constants.g # gravitational acceleration at surface of Earth in m/s^-2
mu_s_dict = {'skin': {'wood': 0.5}} # coefficient of static friction
mu_k_dict = {'skin': {'wood': 0.3}} # coefficient of kinetic friction

class Stick():
    
    def __init__(self, material='wood', len=1, mass=0.2, mu_s=0.5, mu_k=0.3):
        self.material = 'wood'
        self.len = len # in m
        self.mass = mass # in kg

class Finger():

    def __init__(self, material='skin'):
        self.material = 'skin'

class Trick():

    def __init__(
        self, 
        bridge, 
        support1, 
        support2, 
        angle=-15,         
        support1_l0=0, 
        support2_l0=1, 
        u=0.2, 
        dt=0.02, 
        duration=6):
        self.bridge = bridge
        self.support1 = support1
        self.support2 = support2
        if not -90 <= angle <= 90:
            raise RuntimeError('The angle of the bridge must be between -90 and 90 degrees.')        
        self.angle = angle / 360 * 2 * scipy.constants.pi # angle of bridge relative to x-axis
        self.c = bridge.len / 2 # bridge center of mass
        self.c_x = self.c * cos(self.angle)
        # Positions of supports along bridge relative to left-end of bridge (origin)
        if not 0 <= support1_l0 <= self.c:
            raise RuntimeError('The first support must be in the left half of the bridge.')
        if not self.c <= support2_l0 <= bridge.len:
            raise RuntimeError('The second support must be in the right half of the bridge.')            
        self.support1_l = support1_l0
        self.support2_l = support2_l0
        if u <= 0:
            raise RuntimeError('Support sliding speed must be non-negative.')
        self.t = 0
        self.dt = dt # s
        self.dl = u * dt
        self.duration = duration # s

        self.mu_s_1 = mu_s_dict[support1.material][bridge.material]
        self.mu_k_1 = mu_k_dict[support1.material][bridge.material]
        self.mu_s_2 = mu_s_dict[support1.material][bridge.material]
        self.mu_k_2 = mu_k_dict[support1.material][bridge.material]

        self.support1_coords = np.array(
            [[cos(self.angle) * support1_l0], [sin(self.angle) * support1_l0]])
        self.support2_coords = np.array(
            [[cos(self.angle) * support2_l0], [sin(self.angle) * support2_l0]])

        self.force_y = bridge.mass * g
        # force_y = force1_y + force2_y
        # (c_x - support1_x) / (support2_x - c_x) = force1_y / force2_y
        # => (c_x - support1_x / (support2_x - c) = R
        # = force2_y / force1_y = (force_y - force1_y) / force1_y
        # => force1_y = force_y / (R + 1) * force
        force1_y = self.force_y / (
            (self.c_x - self.support1_coords[0]) / (self.support2_coords[0] - self.c_x) + 1
        )
        force2_y = self.force_y - force1_y

        force1_n = force1_y * cos(self.angle)
        force2_n = force2_y * cos(self.angle)

        # Determine if the bridge slides off the supports
        if (
            abs(self.force_y * sin(self.angle)) > 
            self.mu_k_1 * force1_n + self.mu_s_2 * force2_n
        ) or (
            abs(self.force_y * sin(self.angle)) > 
            self.mu_s_1 * force1_n + self.mu_k_2 * force2_n
        ):
            raise RuntimeError('The bridge slid off the supports because the angle was too steep.')

        # Determine which support slides first
        # Prefer the left support in a tie
        if self.mu_s_1 * force1_n <= self.mu_s_2 * force2_n:
            self.slider = self.support1
            self.force1_f = self.mu_k_1 * force1_n
            self.force2_f = self.mu_s_2 * force2_n
        else:
            self.slider = self.support2
            self.force1_f = self.mu_s_1 * force1_n
            self.force2_f = self.mu_k_2 * force2_n

        # Records of distances from supports to center of mass
        self.support1_l_history = [support1_l0 - self.c]
        self.support2_l_history = [support2_l0 - self.c]
        return

    def step(self):
        
        # Both supports should end at the center of gravity without crossing it
        if self.support1_l < self.c and self.support2_l > self.c and self.t < self.duration:

            # The left support was moving
            if self.slider == self.support1:
                support1_coords_t2 = np.array([
                    [self.support1_coords[0] + self.dl * cos(self.angle)], 
                    [self.support1_coords[1] + self.dl * sin(self.angle)]
                ])
                support2_coords_t2 = self.support2_coords
                support1_l_t2 = self.support1_l + self.dl
                support2_l_t2 = self.support2_l

                force1_y_t2 = self.force_y / (
                    (self.c_x - support1_coords_t2[0]) / (support2_coords_t2[0] - self.c_x) + 1
                )
                force2_y_t2 = self.force_y - force1_y_t2
                force1_n_t2 = force1_y_t2 * cos(self.angle)
                force2_n_t2 = force2_y_t2 * cos(self.angle)

                force1_f_t2 = self.mu_k_1 * force1_n_t2
                force2_f_t2 = self.mu_s_2 * force2_n_t2

                # The right support starts moving 
                # when kinetic friction on the left support exceeds static friction on the right
                if self.force2_f + force2_f_t2 < self.force1_f + force1_f_t2:
                    self.slider = self.support2
                    support1_coords_t2 = self.support1_coords
                    support2_coords_t2 = np.array([
                        [self.support2_coords[0] - self.dl * cos(self.angle)], 
                        [self.support2_coords[1] - self.dl * sin(self.angle)]
                    ])
                    support1_l_t2 = self.support1_l
                    support2_l_t2 = self.support2_l - self.dl

                    force1_y_t2 = self.force_y / (
                        (self.c_x - support1_coords_t2[0]) / (support2_coords_t2[0] - self.c_x) + 1
                    )
                    force2_y_t2 = self.force_y - force1_y_t2
                    force1_n_t2 = force1_y_t2 * cos(self.angle)
                    force2_n_t2 = force2_y_t2 * cos(self.angle)  

                    force1_f_t2 = self.mu_s_1 * force1_n_t2
                    force2_f_t2 = self.mu_k_2 * force2_n_t2                              

            # The right support was moving
            else:
                support1_coords_t2 = self.support1_coords
                support2_coords_t2 = np.array([
                    [self.support2_coords[0] - self.dl * cos(self.angle)], 
                    [self.support2_coords[1] - self.dl * sin(self.angle)]
                ])
                support1_l_t2 = self.support1_l
                support2_l_t2 = self.support2_l - self.dl

                force1_y_t2 = self.force_y / (
                    (self.c_x - support1_coords_t2[0]) / (support2_coords_t2[0] - self.c_x) + 1
                )
                force2_y_t2 = self.force_y - force1_y_t2
                force1_n_t2 = force1_y_t2 * cos(self.angle)
                force2_n_t2 = force2_y_t2 * cos(self.angle)

                force1_f_t2 = self.mu_s_1 * force1_n_t2
                force2_f_t2 = self.mu_k_2 * force2_n_t2

                # The left support starts moving 
                # when kinetic friction on the right support exceeds static friction on the left
                if self.force1_f + force1_f_t2 < self.force2_f + force2_f_t2:
                    self.slider = self.support1
                    support1_coords_t2 = np.array([
                        [self.support1_coords[0] + self.dl * cos(self.angle)], 
                        [self.support1_coords[1] + self.dl * sin(self.angle)]
                    ])
                    support2_coords_t2 = self.support2_coords
                    support1_l_t2 = self.support1_l + self.dl
                    support2_l_t2 = self.support2_l

                    force1_y_t2 = self.force_y / (
                        (self.c_x - support1_coords_t2[0]) / (support2_coords_t2[0] - self.c_x) + 1
                    )
                    force2_y_t2 = self.force_y - force1_y_t2
                    force1_n_t2 = force1_y_t2 * cos(self.angle)
                    force2_n_t2 = force2_y_t2 * cos(self.angle)

                    force1_f_t2 = self.mu_k_1 * force1_n_t2
                    force2_f_t2 = self.mu_s_2 * force2_n_t2

            # Update attributes for the next step
            self.t += self.dt
            self.support1_coords = support1_coords_t2
            self.support2_coords = support2_coords_t2
            self.support1_l = support1_l_t2
            self.support2_l = support2_l_t2
            self.force1_f = force1_f_t2
            self.force2_f = force2_f_t2

        self.support1_l_history.append(self.support1_l - self.c)
        self.support2_l_history.append(self.support2_l - self.c)
        return

trick = Trick(Stick(), Finger(), Finger())
fig = plt.figure()
ax1 = fig.add_subplot(
    121, 
    autoscale_on=False, 
    xlim=(0, trick.bridge.len), 
    ylim=(-trick.bridge.len, trick.bridge.len)
)
ax2 = fig.add_subplot(
    122, 
    autoscale_on=False, 
    xlim=(0, trick.duration), 
    ylim=(-trick.bridge.len / 2, trick.bridge.len / 2)
)

ax1.set_aspect('equal')
ax1.set_xlabel('x (m)')
ax1.set_ylabel('y (m)')
support_points, = ax1.plot([], [], color='black', marker='o', markersize=10)
bridge_line, = ax1.plot(
    [0, trick.bridge.len * cos(trick.angle)], 
    [0, trick.bridge.len * sin(trick.angle)], 
    color='white', 
    linewidth=5
)
time_template = 'time = %.1fs'
time_text = ax1.text(0.5, 0.9, '', horizontalalignment='center', transform=ax1.transAxes)

ax2.set_xlabel('time (s)')
ax2.set_ylabel('distance to center of mass (m)')
support1_distances_to_c, = ax2.plot([], [], color='green', linewidth=1)
support2_distances_to_c, = ax2.plot([], [], color='green', linewidth=1)

def init_animation():
    support_points.set_data([], [])
    bridge_line.set_color('white')
    time_text.set_text('')
    support1_distances_to_c.set_data([], [])
    support2_distances_to_c.set_data([], [])
    return support_points, bridge_line, time_text

def animate(i):
    trick.step()

    support_points.set_data(
        [trick.support1_coords[0], trick.support2_coords[0]], 
        [trick.support1_coords[1], trick.support2_coords[1]]
    )
    bridge_line.set_color('blue')
    time_text.set_text(time_template % (i * trick.dt))
    support1_distances_to_c.set_data(
        [trick.dt * i for i in range(len(trick.support1_l_history))], trick.support1_l_history)
    support2_distances_to_c.set_data(
        [trick.dt * i for i in range(len(trick.support2_l_history))], trick.support2_l_history)
    return support_points, bridge_line, time_text, support1_distances_to_c, support2_distances_to_c

ani = animation.FuncAnimation(
    fig, 
    animate, 
    frames=int(trick.duration / trick.dt) + 1, 
    interval=0, 
    blit=True, 
    init_func=init_animation
)

ani.save('friction-trick.mp4', fps=30, extra_args=['-vcodec', 'libx264'])