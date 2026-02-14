
import matplotlib.pyplot as plt
import numpy as np
import torch

from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def render(td, actions=None, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    td = td.detach().cpu()

    if actions is None:
        actions = td.get("action", None)
    
    # if actions is None, try to get from td['action']
    # if not present, just plot locations

    locs = td["locs"]

    x = locs[:, 0]
    y = locs[:, 1]

    # Plot locations
    ax.scatter(x, y, c="blue", label="Locations")
    ax.scatter(x[0], y[0], c="green", s=100, label="Depot (Start)")

    # Annotate order if actions provided
    if actions is not None:
        actions = actions.detach().cpu()
        # Ensure actions is a 1D tensor of indices for a single instance
        if actions.dim() > 1:
             # Take the first in batch if provided batch
             actions = actions[0]
        
        # Plot tour
        tour_locs = locs[actions]
        # Close the loop
        tour_locs = torch.cat((tour_locs, tour_locs[0:1]))
        
        tx = tour_locs[:, 0]
        ty = tour_locs[:, 1]
        
        ax.plot(tx, ty, c="red", linestyle="-", label="Tour")
        
        # Annotate order
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.annotate(str(i), (xi, yi), textcoords="offset points", xytext=(0, 5), ha='center')

    ax.legend()
    return ax
