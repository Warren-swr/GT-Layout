"""
    some utility functions for jupyter notebook visualization
"""

import numpy as np
import matplotlib
# matplotlib.use("Agg")
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

from rand_cmap import rand_cmap
cmap = rand_cmap(300, type='bright', first_color_black=True, last_color_black=False, verbose=False)


def load_semantic_colors(filename):
    semantic_colors = {}
    with open(filename, 'r') as fin:
        for l in fin.readlines():
            _, semantic, r, g, b = l.rstrip().split()
            semantic_colors[semantic] = (int(r), int(g), int(b))
    return semantic_colors

def draw_box(ax, p, color, rot=None):
    x = p[0]# + 0.01
    y = p[1]# + 0.01
    w = p[2]# - 0.02
    h = p[3]# - 0.02

    rect = patches.Rectangle((x, y), w, h, linewidth=3)
    fill_color = (color[0], color[1], color[2], 0.2)
    edge_color = (color[0], color[1], color[2], 1.0)
    rect.set_color(fill_color)
    rect.set_linestyle('-')
    rect.set_edgecolor(edge_color)
    rect.set_fill(True)
    ax.add_patch(rect)


def draw_partnet_objects(objects, object_names=None, figsize=None, out_fn=None, \
        leafs_only=False, use_id_as_color=False, sem_colors_filename=None):
    # load sem colors if provided
    if sem_colors_filename is not None:
        sem_colors = load_semantic_colors(filename=sem_colors_filename)
        for sem in sem_colors:
            sem_colors[sem] = (float(sem_colors[sem][0]) / 255.0, float(sem_colors[sem][1]) / 255.0, float(sem_colors[sem][2]) / 255.0)
    else:
        sem_colors = None

    if figsize is not None:
        fig = plt.figure(0, figsize=figsize)
    else:
        fig = plt.figure(0)
    
    extent = 0.7
    for i, obj in enumerate(objects):
        part_boxes, part_ids, part_sems = obj.get_part_hierarchy(leafs_only=leafs_only, show_mode=True)

        ax = fig.add_subplot(1, len(objects), i+1)
        ax.set_xlim(-0.1, 0.9)
        ax.set_ylim(1.1, -0.1)
        # ax.set_zlim(-extent, extent)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        ax.set_axis_off()
        # ax.set_zlabel('y')
        # ax.set_aspect('auto')
        # ax.set_proj_type('persp')

        if object_names is not None:
            ax.set_title(object_names[i])

        for jj in range(len(part_boxes)):
            if sem_colors is not None:
                color = sem_colors[part_sems[jj]]
            else:
                color_id = part_ids[jj]
                if use_id_as_color:
                    color_id = jj
                color = cmap(color_id)

            if part_boxes[jj] is not None:
                draw_box(ax=ax, p=part_boxes[jj].cpu().numpy().reshape(-1), color=color)

    if out_fn is None:
        # plt.tight_layout()
        plt.show()
    else:
        fig.savefig(out_fn, bbox_inches='tight')
        plt.close(fig)