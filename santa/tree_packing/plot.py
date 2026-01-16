from decimal import Decimal, getcontext

from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from shapely.ops import unary_union

SCALE_FACTOR = Decimal('1e15')

from santa.core import Solution, SolutionEval


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


def plot_solution(solution: Solution | SolutionEval, show_text=False, title=None):
    import matplotlib.pyplot as plt
    import numpy as np

    pos, ang = solution.params

    for i, (pos_i, ang_i) in enumerate(zip(pos, ang)):
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

        plt.plot(x, y)
        plt.fill(x, y, alpha=0.5)
        if show_text:
            plt.text(x_mean, y_mean, "%.3d" % i, color='k', bbox=dict(boxstyle=f"round,pad={0.1}", fc="lightgrey"))

    if title is not None:
        plt.title(title)
    plt.gca().set_aspect('equal')
