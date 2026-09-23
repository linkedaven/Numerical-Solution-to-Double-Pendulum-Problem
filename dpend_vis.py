import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

g = 9.81  # gravity stays fixed

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

def _equations_py(t, y, m1, m2, l1, l2):
    th1, th2, w1, w2 = y
    d = th1 - th2
    cos_d = np.cos(d)
    sin_d = np.sin(d)

    a11 = (m1 + m2) * l1
    a12 = m2 * l2 * cos_d
    a21 = l1 * cos_d
    a22 = l2

    b1 = -m2*l2*w2**2*sin_d - (m1+m2)*g*np.sin(th1)
    b2 = l1*w1**2*sin_d - g*np.sin(th2)

    det = a11*a22 - a12*a21
    a1 = (b1*a22 - a12*b2) / det
    a2 = (a11*b2 - b1*a21) / det

    return np.array([w1, w2, a1, a2])

if NUMBA_AVAILABLE:
    equations = njit(cache=True)(_equations_py)
    equations(0.0, np.array([0.1, 0.1, 0.0, 0.0]), 2.0, 2.0, 1.0, 1.0)
else:
    equations = _equations_py

def simulate(theta1_deg, theta2_deg, omega1, omega2, m1, m2, l1, l2, t_start, t_end, n_points):
    y0 = np.array([np.deg2rad(theta1_deg), np.deg2rad(theta2_deg), omega1, omega2])
    t_span = (t_start, t_end)
    t_eval = np.linspace(t_start, t_end, n_points)

    sol = solve_ivp(
        equations, t_span, y0, t_eval=t_eval,
        args=(m1, m2, l1, l2),
        method='DOP853',
        rtol=1e-8, atol=1e-10
    )
    th1, th2 = sol.y[0], sol.y[1]
    x1 = l1*np.sin(th1)
    y1 = -l1*np.cos(th1)
    x2 = x1 + l2*np.sin(th2)
    y2 = y1 - l2*np.cos(th2)
    return sol.t, th1, th2, x1, y1, x2, y2

# --- Dark theme setup ---
plt.style.use('dark_background')

BG_COLOR = '#1e1e1e'
PANEL_COLOR = '#2b2b2b'
TEXT_COLOR = '#e0e0e0'
GRID_COLOR = '#444444'
ACCENT1 = '#ff9f43'
ACCENT2 = '#54a0ff'
ROD_COLOR = '#aaaaaa'
BUTTON_COLOR = '#3a3a3a'
BUTTON_HOVER = '#505050'

# --- Figure & axes layout ---
fig = plt.figure(figsize=(8, 5))
fig.patch.set_facecolor(BG_COLOR)

ax1    = fig.add_axes([0.07, 0.71, 0.40, 0.24])  # angle vs time (growing)
ax_pend= fig.add_axes([0.55, 0.71, 0.40, 0.24])  # live pendulum swing
ax2    = fig.add_axes([0.07, 0.44, 0.40, 0.22])  # mass 1 trajectory (growing)
ax3    = fig.add_axes([0.55, 0.44, 0.40, 0.22])  # mass 2 trajectory (growing)

def style_axis(ax, title):
    ax.set_facecolor(BG_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5)
    ax.tick_params(colors=TEXT_COLOR, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.set_title(title, fontsize=9)

style_axis(ax1, 'Angular Motion')
ax1.set_xlabel('Time (s)', fontsize=8)
ax1.set_ylabel('Angle (rad)', fontsize=8)

style_axis(ax_pend, 'Live Pendulum')
ax_pend.set_aspect('equal')
ax_pend.set_xticks([]); ax_pend.set_yticks([])

style_axis(ax2, 'Trajectory of Mass 1')
ax2.set_xlabel('x (m)', fontsize=8)
ax2.set_ylabel('y (m)', fontsize=8)
ax2.set_aspect('equal')

style_axis(ax3, 'Trajectory of Mass 2')
ax3.set_xlabel('x (m)', fontsize=8)
ax3.set_ylabel('y (m)', fontsize=8)
ax3.set_aspect('equal')

# --- Line/marker artists (data set on each frame) ---
line_th1, = ax1.plot([], [], color=ACCENT1, linewidth=1, label=r'$\theta_1$')
line_th2, = ax1.plot([], [], color=ACCENT2, linewidth=1, label=r'$\theta_2$')
time_marker = ax1.axvline(0, color=TEXT_COLOR, linewidth=0.8, linestyle='--', alpha=0.6)
leg = ax1.legend(fontsize=7, facecolor=PANEL_COLOR, edgecolor=GRID_COLOR, loc='upper right')
for txt in leg.get_texts():
    txt.set_color(TEXT_COLOR)

rod_line, = ax_pend.plot([], [], color=ROD_COLOR, linewidth=1.5)
trail_line, = ax_pend.plot([], [], color=ACCENT2, linewidth=1, alpha=0.4)
pivot_dot, = ax_pend.plot([0], [0], 'o', color=TEXT_COLOR, markersize=4)
bob1, = ax_pend.plot([], [], 'o', color=ACCENT1, markersize=9)
bob2, = ax_pend.plot([], [], 'o', color=ACCENT2, markersize=9)

trace1, = ax2.plot([], [], color=ACCENT1, linewidth=1)
marker1, = ax2.plot([], [], 'o', color=ACCENT1, markersize=5)
pivot1, = ax2.plot([0], [0], 'o', color=TEXT_COLOR, markersize=4)

trace2, = ax3.plot([], [], color=ACCENT2, linewidth=1)
marker2, = ax3.plot([], [], 'o', color=ACCENT2, markersize=5)
pivot2, = ax3.plot([0], [0], 'o', color=TEXT_COLOR, markersize=4)

all_artists = [line_th1, line_th2, time_marker, rod_line, trail_line,
               bob1, bob2, trace1, marker1, trace2, marker2]

# --- Global state holding the current simulation results ---
current = {}
animation_state = {'step': 0, 'skip': 2, 'paused': False, 'trail_len': 60, 'closed': False}

def recompute_and_reset(theta1_deg, theta2_deg, omega1, omega2, m1, m2, l1, l2,
                         t_start, t_end, n_points, skip):
    if animation_state['closed']:
        return
    t, th1, th2, x1, y1, x2, y2 = simulate(
        theta1_deg, theta2_deg, omega1, omega2, m1, m2, l1, l2, t_start, t_end, n_points)
    current.update(t=t, th1=th1, th2=th2, x1=x1, y1=y1, x2=x2, y2=y2)

    ax1.set_xlim(t_start, t_end)
    pad = 0.1 * (max(th1.max(), th2.max()) - min(th1.min(), th2.min()) + 1e-6)
    ax1.set_ylim(min(th1.min(), th2.min()) - pad, max(th1.max(), th2.max()) + pad)

    reach = (l1 + l2) * 1.15
    ax_pend.set_xlim(-reach, reach)
    ax_pend.set_ylim(-reach, reach)

    for ax, xs, ys in [(ax2, x1, y1), (ax3, x2, y2)]:
        pad_x = 0.15 * (xs.max() - xs.min() + 1e-6)
        pad_y = 0.15 * (ys.max() - ys.min() + 1e-6)
        ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
        ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)

    animation_state['step'] = 0
    animation_state['skip'] = max(1, skip)
    animation_state['trail_len'] = max(10, n_points // 20)

    fig.canvas.draw_idle()

# --- Defaults ---
defaults = {
    'theta1': '120', 'theta2': '20', 'omega1': '0', 'omega2': '0',
    'm1': '2.0', 'm2': '2.0', 'l1': '1.0', 'l2': '1.0',
    't_start': '0', 't_end': '100', 'n_points': '1000', 'speed': '2'
}

recompute_and_reset(float(defaults['theta1']), float(defaults['theta2']),
                     float(defaults['omega1']), float(defaults['omega2']),
                     float(defaults['m1']), float(defaults['m2']),
                     float(defaults['l1']), float(defaults['l2']),
                     float(defaults['t_start']), float(defaults['t_end']),
                     int(defaults['n_points']), int(defaults['speed']))

# --- Animation update function ---
def update(_frame):
    if animation_state['closed'] or animation_state['paused'] or 't' not in current:
        return all_artists

    n = len(current['t'])
    idx = animation_state['step'] % n
    animation_state['step'] += animation_state['skip']

    t, th1, th2 = current['t'], current['th1'], current['th2']
    x1, y1, x2, y2 = current['x1'], current['y1'], current['x2'], current['y2']

    line_th1.set_data(t[:idx+1], th1[:idx+1])
    line_th2.set_data(t[:idx+1], th2[:idx+1])
    time_marker.set_xdata([t[idx], t[idx]])

    rod_line.set_data([0, x1[idx], x2[idx]], [0, y1[idx], y2[idx]])
    bob1.set_data([x1[idx]], [y1[idx]])
    bob2.set_data([x2[idx]], [y2[idx]])
    tstart = max(0, idx - animation_state['trail_len'])
    trail_line.set_data(x2[tstart:idx+1], y2[tstart:idx+1])

    trace1.set_data(x1[:idx+1], y1[:idx+1])
    marker1.set_data([x1[idx]], [y1[idx]])
    trace2.set_data(x2[:idx+1], y2[:idx+1])
    marker2.set_data([x2[idx]], [y2[idx]])

    return all_artists

anim = FuncAnimation(fig, update, interval=20, blit=False, cache_frame_data=False)

# --- Clean shutdown: stop the animation timer the moment the window is closed ---
def on_close(event):
    animation_state['closed'] = True
    animation_state['paused'] = True
    try:
        anim.event_source.stop()
    except Exception:
        pass

fig.canvas.mpl_connect('close_event', on_close)

# --- Text box layout ---
box_w, box_h = 0.10, 0.045
row1_y = 0.32
row2_y = 0.20
row3_y = 0.08

def make_textbox(ax_pos, label, initial):
    ax = fig.add_axes(ax_pos)
    ax.set_facecolor(PANEL_COLOR)
    tb = TextBox(ax, label, initial=initial, color=PANEL_COLOR, hovercolor=BUTTON_HOVER)
    tb.label.set_color(TEXT_COLOR)
    tb.label.set_fontsize(8)
    tb.text_disp.set_color(TEXT_COLOR)
    return tb

tb_th1 = make_textbox([0.08, row1_y, box_w, box_h], r'$\theta_1$ (deg)  ', defaults['theta1'])
tb_th2 = make_textbox([0.26, row1_y, box_w, box_h], r'$\theta_2$ (deg)  ', defaults['theta2'])
tb_w1  = make_textbox([0.44, row1_y, box_w, box_h], r'$\omega_1$ (rad/s)  ', defaults['omega1'])
tb_w2  = make_textbox([0.62, row1_y, box_w, box_h], r'$\omega_2$ (rad/s)  ', defaults['omega2'])

tb_m1 = make_textbox([0.08, row2_y, box_w, box_h], r'$m_1$ (kg)  ', defaults['m1'])
tb_m2 = make_textbox([0.26, row2_y, box_w, box_h], r'$m_2$ (kg)  ', defaults['m2'])
tb_l1 = make_textbox([0.44, row2_y, box_w, box_h], r'$l_1$ (m)  ', defaults['l1'])
tb_l2 = make_textbox([0.62, row2_y, box_w, box_h], r'$l_2$ (m)  ', defaults['l2'])

tb_tstart = make_textbox([0.08, row3_y, box_w, box_h], r'$t_{start}$ (s)  ', defaults['t_start'])
tb_tend   = make_textbox([0.26, row3_y, box_w, box_h], r'$t_{end}$ (s)  ', defaults['t_end'])
tb_npts   = make_textbox([0.44, row3_y, box_w, box_h], r'Points  ', defaults['n_points'])
tb_speed  = make_textbox([0.62, row3_y, box_w, box_h], r'Speed  ', defaults['speed'])

# --- Run button ---
ax_run = fig.add_axes([0.82, row1_y, 0.10, box_h])
ax_run.set_facecolor(BUTTON_COLOR)
btn_run = Button(ax_run, 'Run', color=BUTTON_COLOR, hovercolor=BUTTON_HOVER)
btn_run.label.set_color(TEXT_COLOR)
btn_run.label.set_fontsize(8)

def on_run(event):
    # Guard: ignore stray widget callbacks firing after the window is closing
    if animation_state['closed']:
        return
    try:
        theta1_deg = float(tb_th1.text)
        theta2_deg = float(tb_th2.text)
        omega1 = float(tb_w1.text)
        omega2 = float(tb_w2.text)
        m1 = float(tb_m1.text)
        m2 = float(tb_m2.text)
        l1 = float(tb_l1.text)
        l2 = float(tb_l2.text)
        t_start = float(tb_tstart.text)
        t_end = float(tb_tend.text)
        n_points = int(float(tb_npts.text))
        speed = int(float(tb_speed.text))

        if m1 <= 0 or m2 <= 0 or l1 <= 0 or l2 <= 0:
            print("Masses and lengths must be positive numbers.")
            return
        if t_end <= t_start:
            print("t_end must be greater than t_start.")
            return
        if n_points < 2:
            print("Points must be at least 2.")
            return

        recompute_and_reset(theta1_deg, theta2_deg, omega1, omega2,
                             m1, m2, l1, l2, t_start, t_end, n_points, speed)
    except ValueError:
        print("Please enter valid numbers in all fields.")

btn_run.on_clicked(on_run)

# --- Play/Pause button ---
ax_play = fig.add_axes([0.82, row2_y, 0.10, box_h])
ax_play.set_facecolor(BUTTON_COLOR)
btn_play = Button(ax_play, 'Pause', color=BUTTON_COLOR, hovercolor=BUTTON_HOVER)
btn_play.label.set_color(TEXT_COLOR)
btn_play.label.set_fontsize(8)

def on_play_pause(event):
    if animation_state['closed']:
        return
    animation_state['paused'] = not animation_state['paused']
    btn_play.label.set_text('Play' if animation_state['paused'] else 'Pause')
    fig.canvas.draw_idle()

btn_play.on_clicked(on_play_pause)

if not NUMBA_AVAILABLE:
    print("Tip: install numba ('pip install numba') for faster simulation.")

plt.show()