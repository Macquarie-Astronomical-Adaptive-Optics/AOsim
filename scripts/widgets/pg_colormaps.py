import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg

def mpl_cmap_to_pg(name, n=256):
    cmap = plt.get_cmap(name)
    colors = (cmap(np.linspace(0, 1, n))[:, :3] * 255).astype(np.ubyte)
    pos = np.linspace(0, 1, n)
    return pg.ColorMap(pos, colors)

def apply_mpl_cmap(image_item, img, cmap="viridis", vmin=None, vmax=None):
    if img.size == 0:
        image_item.clear()
        return

    if vmin is None or vmax is None:
        finite = np.isfinite(img)
        if not np.any(finite):
            vmin, vmax = 0.0, 1.0
        else:
            if vmin is None:
                vmin = img[finite].min()
            if vmax is None:
                vmax = img[finite].max()

    if vmin == vmax:
        eps = 1e-12 if vmin == 0 else abs(vmin) * 1e-6
        vmin -= eps
        vmax += eps

    image_item.setImage(
        img,
        levels=(vmin, vmax),
        autoLevels=False
    )

    pg_cmap = mpl_cmap_to_pg(cmap)
    image_item.setLookupTable(pg_cmap.getLookupTable())
