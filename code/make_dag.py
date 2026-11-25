"""Fig. A1: assumed relations between instruction, co-occurring topics, response style and endorsement."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

W, H = 0.20, 0.12
N = {'X': (0.12, 0.88, 'Background (ESCS, gender,\ngrade, age, test effort)'),
     'S': (0.12, 0.12, 'System\n(fixed effects)'),
     'R': (0.50, 0.88, 'Reading\nachievement'),
     'A': (0.88, 0.88, 'Response style\n(agreement index)'),
     'O': (0.12, 0.50, 'Other instruction\ntopics (six)'),
     'T': (0.50, 0.50, 'Phishing instruction\n(self-report)'),
     'Y': (0.88, 0.50, 'Endorsement of\nthe phishing link'),
     'E': (0.88, 0.12, 'Selection by\noutcome-defined rules')}
E = [('X', 'R'), ('X', 'O'), ('X', 'T'), ('S', 'O'), ('S', 'T'), ('S', 'Y'),
     ('R', 'T'), ('R', 'Y'), ('A', 'T'), ('A', 'Y'), ('T', 'Y'), ('Y', 'E')]


def clip(x0, y0, x1, y1, pad=0.012):
    """Point where the segment from box centre (x0, y0) towards (x1, y1) leaves the box."""
    dx, dy = x1 - x0, y1 - y0
    t = min((W / 2 + pad) / abs(dx) if dx else 9, (H / 2 + pad) / abs(dy) if dy else 9)
    return x0 + t * dx, y0 + t * dy


fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.set_xlim(0, 1); ax.set_ylim(0.03, 0.97); ax.axis('off')
for k, (x, y, lab) in N.items():
    fc = '#fde0dd' if k == 'E' else ('#deebf7' if k in ('T', 'Y') else '#f2f2f2')
    ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle='round,pad=0.005', fc=fc, ec='#444', lw=0.8))
    ax.text(x, y, lab, ha='center', va='center', fontsize=7.5)


def arrow(a, b, color='#777', lw=0.9, ls='-', style='-|>'):
    xa, ya, _ = N[a]; xb, yb, _ = N[b]
    p0 = clip(xa, ya, xb, yb); p1 = clip(xb, yb, xa, ya)
    ax.annotate('', xy=p1, xytext=p0, arrowprops=dict(arrowstyle=style, color=color, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


for a, b in E:
    if (a, b) == ('T', 'Y'):
        arrow(a, b, '#1f4e79', 1.8)
    elif b == 'E':
        arrow(a, b, '#c0392b', 1.2, '--')
    else:
        arrow(a, b)
arrow('O', 'T', '#777', 0.9, ':', '<|-|>')
ax.annotate('', xy=(0.83, 0.50 - H / 2 - 0.012), xytext=(0.17, 0.50 - H / 2 - 0.012),
            arrowprops=dict(arrowstyle='-|>', color='#777', lw=0.9, shrinkA=0, shrinkB=0, connectionstyle='arc3,rad=0.25'))
ax.text(0.31, 0.535, 'shared curriculum\nor reporting', ha='center', va='bottom', fontsize=6.3, color='#555')
fig.tight_layout(); fig.savefig('manuscript/figA1_dag.png', dpi=300)
