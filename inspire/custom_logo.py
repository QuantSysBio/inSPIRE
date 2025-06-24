""" Functions to allow custom logo plotting.
"""
from logomaker import Logo
from logomaker.src.colors import get_color_dict
from logomaker.src.Glyph import Glyph
from logomaker.src.matrix import transform_matrix
from logomaker.src.error_handling import handle_errors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class CustomLogo(Logo):
    """ Custom class to allow individually set alphas.
    """
    @handle_errors
    def __init__(self,
                 df,
                 color_scheme=None,
                 font_name='sans',
                 stack_order='big_on_top',
                 center_values=False,
                 baseline_width=0.5,
                 flip_below=False,
                 shade_below=0.0,
                 fade_below=0.0,
                 fade_probabilities=False,
                 vpad=0.0,
                 vsep=0.0,
                 alpha=1.0,
                 show_spines=None,
                 ax=None,
                 zorder=0,
                 figsize=(10, 2.5),
                 custom_alphas=None,
                 **kwargs):
        """Modified init function."""
        # set class attributes
        self.df = df
        self.color_scheme = color_scheme
        self.font_name = font_name
        self.stack_order = stack_order
        self.center_values = center_values
        self.baseline_width = baseline_width
        self.flip_below = flip_below
        self.shade_below = shade_below
        self.fade_below = fade_below
        self.fade_probabilities = fade_probabilities
        self.vpad = vpad
        self.vsep = vsep
        self.alpha = alpha
        self.show_spines = show_spines
        self.zorder = zorder
        self.figsize = figsize
        self.ax = ax

        # save other keyword arguments
        self.glyph_kwargs = kwargs

        # register logo as NOT having been drawn
        # This is changed to True after all Glyphs have been rendered
        self.has_been_drawn = False

        # perform input checks to validate attributes
        self._input_checks()

        # compute length
        self.L = len(self.df)

        # get list of characters
        self.cs = np.array([c for c in self.df.columns])
        self.C = len(self.cs)

        # get color dictionary
        # NOTE: this validates color_scheme; not self._input_checks()
        self.rgb_dict = get_color_dict(self.color_scheme, self.cs)

        # get list of positions
        self.ps = np.array([p for p in self.df.index])

        # center matrix if requested
        if self.center_values:
            self.df = transform_matrix(self.df, center_values=True)

        # create axes if not specified by user
        if self.ax is None:
            _, ax = plt.subplots(1, 1, figsize=self.figsize)
            self.ax = ax

        # save figure as attribute
        self.fig = ax.figure

        # compute characters
        if custom_alphas is None:
            self._compute_glyphs()
        else:
            self._compute_custom_alpha_glyphs(custom_alphas)

        # style glyphs below x-axis
        self.style_glyphs_below(shade=self.shade_below,
                                fade=self.fade_below,
                                flip=self.flip_below)

        # fade glyphs by value if requested
        if self.fade_probabilities:
            self.fade_glyphs_in_probability_logo(v_alpha0=0,
                                                 v_alpha1=1)

        # draw
        self.draw()

    def _compute_custom_alpha_glyphs(self, custom_alphas):
        """
        Specifies the placement and styling of all glyphs within the logo.
        """
        # Create a dataframe of glyphs
        glyph_df = pd.DataFrame()

        # For each position
        for p in self.ps:

            # get values at this position
            vs = np.array(self.df.loc[p, :])

            # Sort values according to the order in which the user
            # wishes the characters to be stacked
            if self.stack_order == 'big_on_top':
                ordered_indices = np.argsort(vs)

            elif self.stack_order == 'small_on_top':
                tmp_vs = np.zeros(len(vs))
                indices = (vs != 0)
                tmp_vs[indices] = 1.0/vs[indices]
                ordered_indices = np.argsort(tmp_vs)

            elif self.stack_order == 'fixed':
                ordered_indices = np.array(range(len(vs)))[::-1]

            else:
                assert False, 'This line of code should never be called.'

            # Reorder values and characters
            vs = vs[ordered_indices]
            cs = [str(c) for c in self.cs[ordered_indices]]

            # Set floor
            floor = sum((vs - self.vsep) * (vs < 0)) + self.vsep/2.0

            # For each character
            for v, c in zip(vs, cs):

                # Set ceiling
                ceiling = floor + abs(v)

                # Set color
                this_color = self.rgb_dict[c]

                # Set whether to flip character
                flip = (v < 0 and self.flip_below)

                # Create glyph if height is finite
                glyph = Glyph(p, c,
                              ax=self.ax,
                              floor=floor,
                              ceiling=ceiling,
                              color=this_color,
                              flip=flip,
                              zorder=self.zorder,
                              font_name=self.font_name,
                              alpha=custom_alphas[f'{c}{p}'],
                              vpad=self.vpad,
                              **self.glyph_kwargs)

                # Add glyph to glyph_df
                glyph_df.loc[p, c] = glyph

                # Raise floor to current ceiling
                floor = ceiling + self.vsep

        # Set glyph_df attribute
        self.glyph_df = glyph_df
        self.glyph_list = [g for g in self.glyph_df.values.ravel()
                           if isinstance(g, Glyph)]
