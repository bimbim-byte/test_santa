# packing_improved.py
import math
from decimal import Decimal, getcontext
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from shapely import affinity
from shapely.ops import unary_union
from matplotlib.patches import Rectangle
import time
import os

os.makedirs("output", exist_ok=True)

# --------------------------
# Precision & scale
# --------------------------
getcontext().prec = 25
scale_factor = Decimal('1e15')

# --------------------------
# Base ChristmasTree (same shape as original)
# --------------------------
class ChristmasTree:
    def __init__(self, center_x='0', center_y='0', angle=0):
        self.center_x = Decimal(str(center_x))
        self.center_y = Decimal(str(center_y))
        self.angle = int(angle)  # use integer degrees (0,90,180,270)

        trunk_w = Decimal('0.15'); trunk_h = Decimal('0.2')
        base_w = Decimal('0.7'); mid_w = Decimal('0.4'); top_w = Decimal('0.25')
        tip_y = Decimal('0.8'); tier_1_y = Decimal('0.5'); tier_2_y = Decimal('0.25'); base_y = Decimal('0.0')
        trunk_bottom_y = -trunk_h

        coords = [
            (Decimal('0.0'), tip_y),
            (top_w / 2, tier_1_y), (top_w / 4, tier_1_y),
            (mid_w / 2, tier_2_y), (mid_w / 4, tier_2_y),
            (base_w / 2, base_y), (trunk_w / 2, base_y), (trunk_w / 2, trunk_bottom_y),
            (-trunk_w / 2, trunk_bottom_y), (-trunk_w / 2, base_y), (-base_w / 2, base_y),
            (-mid_w / 4, tier_2_y), (-mid_w / 2, tier_2_y),
            (-top_w / 4, tier_1_y), (-top_w / 2, tier_1_y)
        ]

        scaled_coords = [(float(x * scale_factor), float(y * scale_factor)) for x, y in coords]
        initial_polygon = Polygon(scaled_coords)

        # rotate around origin then translate
        rotated = affinity.rotate(initial_polygon, self.angle, origin=(0, 0))
        self.polygon = affinity.translate(rotated,
                                          xoff=float(self.center_x * scale_factor),
                                          yoff=float(self.center_y * scale_factor))

    def translated(self, dx, dy):
        """Return a new ChristmasTree instance translated by dx,dy (Decimal)"""
        new = ChristmasTree(self.center_x, self.center_y, self.angle)
        new.polygon = affinity.translate(self.polygon, xoff=float(dx * scale_factor), yoff=float(dy * scale_factor))
        new.center_x = self.center_x + Decimal(str(dx))
        new.center_y = self.center_y + Decimal(str(dy))
        return new

    def move_by(self, dx, dy):
        """Mutate by moving"""
        self.polygon = affinity.translate(self.polygon, xoff=float(dx * scale_factor), yoff=float(dy * scale_factor))
        self.center_x += Decimal(str(dx))
        self.center_y += Decimal(str(dy))

# --------------------------
# Utilities
# --------------------------
def get_polygon_size(poly):
    """Return (width, height) in unscaled Decimal units"""
    b = poly.bounds  # (minx, miny, maxx, maxy) in scaled floats
    minx, miny, maxx, maxy = [Decimal(str(v)) for v in b]
    w = (maxx - minx) / scale_factor
    h = (maxy - miny) / scale_factor
    return float(w), float(h)

def bounds_decimal_from_union(polys):
    u = unary_union(polys)
    b = u.bounds
    return [Decimal(str(v)) / scale_factor for v in b]  # minx,miny,maxx,maxy

# --------------------------
# Create mixed-lattice with rot & hex-like offset
# --------------------------
def create_improved_lattice(n_trees, base_params):
    """
    base_params: dict with keys: gap_x, gap_y, base_y_shift (fraction), hex_y_frac, hex_x_frac
    Returns list of ChristmasTree objects
    """
    gap_x = float(base_params.get('gap_x', 0.63110701))
    gap_y = float(base_params.get('gap_y', 0.66383987))
    base_y_shift = float(base_params.get('y_shift', 0.44429069))
    hex_y_frac = float(base_params.get('hex_y_frac', 0.5))  # fraction of gap_y to offset for hex packing
    hex_x_frac = float(base_params.get('hex_x_frac', 0.5))  # fraction of gap_x for additional small shift

    # determine grid columns (keep near-square)
    n_cols = int(math.ceil(math.sqrt(n_trees)))
    trees = []
    count = 0
    row = 0

    # We'll pick rotation pattern depending on row parity and column to better fill gaps.
    # pattern choices = [0, 90, 180, 270]
    rotations = [0, 90, 180, 270]

    # Pre-generate a prototype tree so we can compute its approximate dimension for adaptive indent
    proto = ChristmasTree(0, 0, 0)
    proto_w, proto_h = get_polygon_size(proto.polygon)

    # Adaptive row indent based on proto width (keeps rows tighter)
    adaptive_row_indent = min(proto_w * 0.25, 0.3 * gap_x)

    while count < n_trees:
        current_y = row * gap_y
        for col in range(n_cols):
            if count >= n_trees:
                break
            # base x coordinate
            current_x = col * gap_x

            # hex-like column offset (shift every other column vertically)
            if col % 2 == 1:
                current_y_eff = current_y + hex_y_frac * gap_y
            else:
                current_y_eff = current_y

            # row-based indent (offset alternate rows)
            x_eff = current_x
            if row % 2 == 1:
                x_eff += adaptive_row_indent

            # small additional column-dependent x offset to break regular holes
            x_eff += (col % 3) * (hex_x_frac * gap_x * 0.06)

            # rotation strategy:
            # - even rows: alternate 0/180 to interlock vertically
            # - odd rows: use 90/270 to fill horizontal gaps
            if row % 2 == 0:
                angle = 0 if (col % 2 == 0) else 180
            else:
                angle = 90 if (col % 2 == 0) else 270

            # slight y shift for rotated pieces (help interlock)
            y_eff = current_y_eff
            if angle in (180, 270):
                y_eff += base_y_shift * (0.8 if angle == 180 else 0.5)

            trees.append(ChristmasTree(center_x=str(x_eff), center_y=str(y_eff), angle=angle))
            count += 1
        row += 1

    return trees

# --------------------------
# Post-processing compaction
# --------------------------
def compact_trees_greedy(trees, epsilon=1e-4, max_iters=200):
    """
    Greedy compaction:
    1) shift everything left until something touches
    2) shift everything down until something touches
    3) then for each tree try to move left in small steps until collision
    Repeat a few times to settle.
    epsilon: step size in Decimal units
    """
    # Convert epsilon to Decimal
    eps = Decimal(str(epsilon))

    # Step A: translate so minx,miny = 0,0 (anchor)
    polys = [t.polygon for t in trees]
    minx, miny, maxx, maxy = [Decimal(str(v))/scale_factor for v in unary_union(polys).bounds]
    shiftx = -minx
    shifty = -miny
    for t in trees:
        t.move_by(shiftx, shifty)

    # Iteratively compress horizontally then vertically
    for it in range(max_iters):
        moved_any = False
        # Horizontal compaction greedy: try each tree in random order for better packing
        indices = list(range(len(trees)))
        np.random.shuffle(indices)
        for i in indices:
            t = trees[i]
            # attempt to move left in decreasing step sizes (binary-like)
            step = Decimal('0.02')  # start with bigger step
            while step >= eps:
                # test movement
                try_move = step * Decimal('-1')  # negative = left
                new_poly = affinity.translate(t.polygon, xoff=float(try_move * scale_factor), yoff=0.0)
                collision = False
                for j, other in enumerate(trees):
                    if j == i: continue
                    if new_poly.intersects(other.polygon):
                        collision = True
                        break
                if collision:
                    step = step / Decimal('2')
                else:
                    # commit move
                    t.move_by(try_move, Decimal('0'))
                    moved_any = True
                    # keep trying same step (might be able to move further)
                # safety guard
                if step < eps:
                    break

        # Vertical compaction greedy (downwards)
        indices = list(range(len(trees)))
        np.random.shuffle(indices)
        for i in indices:
            t = trees[i]
            step = Decimal('0.02')
            while step >= eps:
                try_move = step * Decimal('-1')
                new_poly = affinity.translate(t.polygon, xoff=0.0, yoff=float(try_move * scale_factor))
                collision = False
                for j, other in enumerate(trees):
                    if j == i: continue
                    if new_poly.intersects(other.polygon):
                        collision = True
                        break
                if collision:
                    step = step / Decimal('2')
                else:
                    t.move_by(Decimal('0'), try_move)
                    moved_any = True
                if step < eps:
                    break

        if not moved_any:
            break

    # Final anchor: move minx,miny to 0,0 again
    polys = [t.polygon for t in trees]
    minx, miny, maxx, maxy = [Decimal(str(v))/scale_factor for v in unary_union(polys).bounds]
    shiftx = -minx
    shifty = -miny
    for t in trees:
        t.move_by(shiftx, shifty)

    return trees

# --------------------------
# Plotting (improved)
# --------------------------
def plot_packing(n, trees, side_length, title_extra='', save_path=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(trees)))
    for i, tree in enumerate(trees):
        x_scaled, y_scaled = tree.polygon.exterior.xy
        x = [Decimal(str(val)) / scale_factor for val in x_scaled]
        y = [Decimal(str(val)) / scale_factor for val in y_scaled]
        ax.fill([float(xx) for xx in x], [float(yy) for yy in y], alpha=0.6, color=colors[i])
        ax.plot([float(xx) for xx in x], [float(yy) for yy in y], color='k', linewidth=0.2, alpha=0.3)

    u_poly = unary_union([t.polygon for t in trees])
    bounds = u_poly.bounds
    minx, miny, maxx, maxy = [Decimal(str(b)) / scale_factor for b in bounds]
    rect = Rectangle((float(minx), float(miny)), float(side_length), float(side_length),
                     fill=False, edgecolor='red', linewidth=2, linestyle='--')
    ax.add_patch(rect)

    ax.set_aspect('equal')
    ax.set_title(f'N={n} | Side: {side_length:.6f} {title_extra}')
    plt.axis('off')

    # <<--- Tambahan penting untuk PNG
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close(fig)



# --------------------------
# Main loop: 1..200 (like original)
# --------------------------
def main():
    print("🔧 Running improved packing (1..200). This may take a moment...)")
    BASE_PARAMS = {
        'gap_x': 0.60,      # slightly tuned
        'gap_y': 0.62,
        'y_shift': 0.40,
        'hex_y_frac': 0.5,
        'hex_x_frac': 0.5
    }

    submission_data = []
    t0 = time.time()
    for n in range(1, 201):
        trees = create_improved_lattice(n, BASE_PARAMS)

        # post-process compaction to squeeze the layout
        trees = compact_trees_greedy(trees, epsilon=1e-4, max_iters=120)

        # compute bounding side
        u_poly = unary_union([t.polygon for t in trees])
        bounds = u_poly.bounds
        w = (Decimal(str(bounds[2])) - Decimal(str(bounds[0]))) / scale_factor
        h = (Decimal(str(bounds[3])) - Decimal(str(bounds[1]))) / scale_factor
        side_length = max(w, h)

        # store submission rows
        for i, tree in enumerate(trees):
            tree_id = f'{n:03d}_{i}'
            submission_data.append([tree_id, tree.center_x, tree.center_y, tree.angle])

        # every 10 show plot
        if n % 10 == 0:
            elapsed = time.time() - t0
            print(f"[{n}/200] Side Length: {side_length:.6f} (elapsed {elapsed:.1f}s)")
            save_name = f"output/packing_{n:03d}.png"
            plot_packing(n, trees, float(side_length), title_extra="(improved packing)", save_path=save_name)


    # Save CSV similar to original: round to 6 decimals and prefix 's'
    df = pd.DataFrame(submission_data, columns=['id', 'x', 'y', 'deg'])
    for col in ['x', 'y', 'deg']:
        df[col] = df[col].astype(float).round(6)
    for col in ['x', 'y', 'deg']:
        df[col] = 's' + df[col].astype('string')
    df.to_csv('submission_improved.csv', index=False)
    print("✅ Done. 'submission_improved.csv' saved.")

if __name__ == '__main__':
    main()
