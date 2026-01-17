import jax.numpy as jnp

from decimal import Decimal, getcontext

from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.ops import unary_union

SCALE_FACTOR = Decimal('1e15')

from santa.core import Solution, SolutionEval
from santa.tree_packing import tree as tree_


class ChristmasTree:
    """Represents a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x='0', center_y='0', angle='0', scale_factor='1e15'):
        """Initializes the Christmas tree with a specific position and rotation."""
        getcontext().prec = 25

        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)
        self.scale_factor = Decimal(scale_factor)
        scale_factor = self.scale_factor

        trunk_w = Decimal('0.15')
        trunk_h = Decimal('0.2')
        base_w = Decimal('0.7')
        mid_w = Decimal('0.4')
        top_w = Decimal('0.25')
        tip_y = Decimal('0.8')
        tier_1_y = Decimal('0.5')
        tier_2_y = Decimal('0.25')
        base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at Tip
                (Decimal('0.0') * scale_factor, tip_y * scale_factor),
                # Right side - Top Tier
                (top_w / Decimal('2') * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal('4') * scale_factor, tier_1_y * scale_factor),
                # Right side - Middle Tier
                (mid_w / Decimal('2') * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal('4') * scale_factor, tier_2_y * scale_factor),
                # Right side - Bottom Tier
                (base_w / Decimal('2') * scale_factor, base_y * scale_factor),
                # Right Trunk
                (trunk_w / Decimal('2') * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal('2') * scale_factor, trunk_bottom_y * scale_factor),
                # Left Trunk
                (-(trunk_w / Decimal('2')) * scale_factor, trunk_bottom_y * scale_factor),
                (-(trunk_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Bottom Tier
                (-(base_w / Decimal('2')) * scale_factor, base_y * scale_factor),
                # Left side - Middle Tier
                (-(mid_w / Decimal('4')) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal('2')) * scale_factor, tier_2_y * scale_factor),
                # Left side - Top Tier
                (-(top_w / Decimal('4')) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal('2')) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * scale_factor),
                                          yoff=float(self.center_y * scale_factor))

    def get_params(self):
        return self.center_x, self.center_y, self.angle


def plot_solution(
        solution: Solution | SolutionEval,
        show_text=False,
        title=None,
        show_grid=False,
        show_center_circle=False,
        grid_mask=None,
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    pos, ang = solution.params

    aux_data = solution.aux_data
    grid = aux_data.get("grid") if aux_data is not None else None

    min_x = None
    max_x = None
    min_y = None
    max_y = None

    mask_coords = None
    mask_values = None
    if grid is not None:
        n = int(float(grid.n))
        size = float(grid.size)
        center = float(grid.center)
        n_cells = n
        n_half = n // 2
        if grid_mask is not None:
            mask_values = np.asarray(grid_mask).T
            if mask_values.ndim != 2:
                raise ValueError("grid_mask must be a 2D array")
            if mask_values.shape == (n, n):
                n_cells = n
                n_half = n // 2
            elif mask_values.shape == (n + 2, n + 2):
                n_cells = n + 2
                n_half = n // 2 + 1
            else:
                raise ValueError(
                    f"grid_mask shape must be {(n, n)} or {(n + 2, n + 2)}, got {mask_values.shape}"
                )
        mask_coords = [center + (i - n_half) * size for i in range(n_cells + 1)]

    if mask_coords is not None and mask_values is not None:
        ax = plt.gca()
        ax.pcolormesh(
            mask_coords,
            mask_coords,
            mask_values,
            cmap="Blues",
            shading="auto",
            alpha=0.3,
            zorder=-2,
        )

    if show_grid:
        if mask_coords is not None:
            ax = plt.gca()
            for coord in mask_coords:
                ax.axhline(coord, color="lightgrey", linewidth=0.5, zorder=-1)
                ax.axvline(coord, color="lightgrey", linewidth=0.5, zorder=-1)

    if show_center_circle:
        ax = plt.gca()
        centers = tree_.get_tree_centers(solution.params)
        radius = float(tree_.THR) / 2.0
        for i, center in enumerate(centers):
            if jnp.isnan(center).sum() > 0:
                continue
            cx = float(center[0])
            cy = float(center[1])
            circle = patches.Circle(
                (cx, cy),
                radius,
                fill=False,
                edgecolor=f"C{i}",
                linewidth=1.0,
                zorder=1,
            )
            ax.add_patch(circle)
            min_x = cx - radius if min_x is None else min(min_x, cx - radius)
            max_x = cx + radius if max_x is None else max(max_x, cx + radius)
            min_y = cy - radius if min_y is None else min(min_y, cy - radius)
            max_y = cy + radius if max_y is None else max(max_y, cy + radius)

    for i, (pos_i, ang_i) in enumerate(zip(pos, ang)):
        if jnp.isnan(pos_i).sum() > 0:
            continue
        # Rescale for plotting
        tree = ChristmasTree(
            center_x=str(float(pos_i[0])),
            center_y=str(float(pos_i[1])),
            angle=str(float(ang_i / np.pi * 180.))
        )

        x_scaled, y_scaled = tree.polygon.exterior.xy
        x = [Decimal(val) / tree.scale_factor for val in x_scaled]
        y = [Decimal(val) / tree.scale_factor for val in y_scaled]

        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))

        plt.plot(x, y, color=f"C{i}")
        plt.fill(x, y, color=f"C{i}", alpha=0.5)
        if show_text:
            plt.text(x_mean, y_mean, "%.3d" % i, color='k', bbox=dict(boxstyle=f"round,pad={0.1}", fc="lightgrey"))
        x_min_i = float(min(x))
        x_max_i = float(max(x))
        y_min_i = float(min(y))
        y_max_i = float(max(y))
        min_x = x_min_i if min_x is None else min(min_x, x_min_i)
        max_x = x_max_i if max_x is None else max(max_x, x_max_i)
        min_y = y_min_i if min_y is None else min(min_y, y_min_i)
        max_y = y_max_i if max_y is None else max(max_y, y_max_i)

    if title is not None:
        plt.title(title)
    if min_x is None and mask_coords is not None:
        min_x = min(mask_coords)
        max_x = max(mask_coords)
        min_y = min(mask_coords)
        max_y = max(mask_coords)
    if min_x is not None and min_y is not None:
        max_abs = max(abs(min_x), abs(max_x), abs(min_y), abs(max_y))
        ax = plt.gca()
        ax.set_xlim(-max_abs, max_abs)
        ax.set_ylim(-max_abs, max_abs)
    plt.gca().set_aspect('equal')
