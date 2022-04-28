#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pymatgen periodic table

import math
from typing import Literal

import numpy as np
from matplotlib import cm, colors

from pymatgen.core.periodic_table import Element

def _decide_fontcolor(rgba: tuple) -> Literal["black", "white"]:
    red, green, blue, _ = rgba
    if (red * 0.299 + green * 0.587 + blue * 0.114) * 255 > 186:
        return "black"

    return "white"

def periodic_table_heatmap(
    elemental_data,
    cbar_label="",
    cbar_label_size=14,
    show_plot=False,
    cmap="YlOrRd",
    cmap_range=None,
    blank_color="grey",
    edge_color="white",
    value_format=None,
    value_fontsize=10,
    symbol_fontsize=14,
    max_row=9,
    readable_fontcolor=False,
):
    """
    A static method that generates a heat map overlaid on a periodic table.
    Args:
         elemental_data (dict): A dictionary with the element as a key and a
            value assigned to it, e.g. surface energy and frequency, etc.
            Elements missing in the elemental_data will be grey by default
            in the final table elemental_data={"Fe": 4.2, "O": 5.0}.
         cbar_label (string): Label of the colorbar. Default is "".
         cbar_label_size (float): Font size for the colorbar label. Default is 14.
         cmap_range (tuple): Minimum and maximum value of the colormap scale.
            If None, the colormap will automatically scale to the range of the
            data.
         show_plot (bool): Whether to show the heatmap. Default is False.
         value_format (str): Formatting string to show values. If None, no value
            is shown. Example: "%.4f" shows float to four decimals.
         value_fontsize (float): Font size for values. Default is 10.
         symbol_fontsize (float): Font size for element symbols. Default is 14.
         cmap (string): Color scheme of the heatmap. Default is 'YlOrRd'.
            Refer to the matplotlib documentation for other options.
         blank_color (string): Color assigned for the missing elements in
            elemental_data. Default is "grey".
         edge_color (string): Color assigned for the edge of elements in the
            periodic table. Default is "white".
         max_row (integer): Maximum number of rows of the periodic table to be
            shown. Default is 9, which means the periodic table heat map covers
            the standard 7 rows of the periodic table + 2 rows for the lanthanides
            and actinides. Use a value of max_row = 7 to exclude the lanthanides and
            actinides.
         readable_fontcolor (bool): Whether to use readable fontcolor depending
            on background color. Default is False.
    """

    # Convert primitive_elemental data in the form of numpy array for plotting.
    if cmap_range is not None:
        max_val = cmap_range[1]
        min_val = cmap_range[0]
    else:
        max_val = max(elemental_data.values())
        min_val = min(elemental_data.values())

    max_row = min(max_row, 9)

    if max_row <= 0:
        raise ValueError("The input argument 'max_row' must be positive!")

    value_table = np.empty((max_row, 18)) * np.nan
    blank_value = min_val - 0.01

    for el in Element:
        value = elemental_data.get(el.symbol, blank_value)
        if 57 <= el.Z <= 71:
            plot_row = 8
            plot_group = (el.Z - 54) % 32
        elif 89 <= el.Z <= 103:
            plot_row = 9
            plot_group = (el.Z - 54) % 32
        else:
            plot_row = el.row
            plot_group = el.group
        if plot_row > max_row:
            continue
        value_table[plot_row - 1, plot_group - 1] = value

    # Initialize the plt object
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(12, 4.5)

    # We set nan type values to masked values (ie blank spaces)
    data_mask = np.ma.masked_invalid(value_table.tolist())
    heatmap = ax.pcolor(
        data_mask,
        cmap=cmap,
        edgecolors=edge_color,
        linewidths=1,
        vmin=min_val - 0.001,
        vmax=max_val + 0.001,
    )
    cbar = fig.colorbar(heatmap)

    # Grey out missing elements in input data
    cbar.cmap.set_under(blank_color)

    # Set the colorbar label and tick marks
    cbar.set_label(cbar_label, rotation=270, labelpad=25, size=cbar_label_size, fontname= 'Arial')
    cbar.ax.tick_params(labelsize=cbar_label_size)

    # Refine and make the table look nice
    ax.axis("off")
    ax.invert_yaxis()

    # Set the scalermap for fontcolor
    norm = colors.Normalize(vmin=min_val, vmax=max_val)
    scalar_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)

    # Label each block with corresponding element and value
    for i, row in enumerate(value_table):
        for j, el in enumerate(row):
            if not np.isnan(el):
                symbol = Element.from_row_and_group(i + 1, j + 1).symbol
                rgba = scalar_cmap.to_rgba(el)
                fontcolor = _decide_fontcolor(rgba) if readable_fontcolor else "black"
                plt.text(
                    j + 0.5,
                    i + 0.4,
                    symbol,
                    horizontalalignment="center",
                    verticalalignment="center",
                    fontname= 'Arial',
                    fontsize=symbol_fontsize,
                    color=fontcolor,
                )
                if el != blank_value and value_format is not None:
                    plt.text(
                        j + 0.5,
                        i + 0.75,
                        value_format % el,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontname= 'Arial',
                        fontsize=value_fontsize,
                        color=fontcolor,
                    )

    plt.tight_layout()

    if show_plot:
        plt.show()

    return plt

