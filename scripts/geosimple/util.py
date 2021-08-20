from typing import Dict, Any, Callable
from geoviz.builder import PlotBuilder
from geoviz.utils.util import parse_args_kwargs
import json

fn_map: Dict[str, Callable] = {
    'remove empties': PlotBuilder.remove_empties,
    'logarithmic scale': PlotBuilder.logify_scale,
    'generate grid': PlotBuilder.auto_grid,
    'simple clip': PlotBuilder.simple_clip,
    'clip datasets': PlotBuilder.clip_datasets,
    'discrete scale': PlotBuilder.discretize_scale,
}

output_fns = ['display figure', 'output figure']


def run_json(filepath: str, debug: bool = False):
    """Runs a json file representation of a plot scheme.

    :param filepath: The filepath to the json file
    :type filepath: str
    :param debug: Determines if parts of the internal processes are printed or not
    :type debug: bool
    :return: The status of the plot after building
    :rtype: PlotStatus
    """
    with open(filepath) as jse:
        read = json.load(jse)

    build_args = {
        "raise_errors": False,
        "plot_regions": True,
        "plot_grids": True,
        "plot_outlines": True,
        "plot_points": True
    }

    if debug:
        debugprint = lambda x: print(x)
    else:
        debugprint = lambda x: None

    mapbox_fig = read.pop("mapbox_token", False)
    adjustments = read.pop("adjustments", {})
    adjust_opacity = adjustments.pop("opacity", True)
    adjust_colorbar = adjustments.pop("colorbar_size", True)
    adjust_focus = adjustments.pop("focus", True)
    build_args.update(adjustments.pop("build", {}))
    output_fig = read.pop("output_figure", False)
    display_fig = read.pop("display_figure", True)
    builder_fns = read.pop("builder_functions", {})

    builder = PlotBuilder.builder_from_dict(**read)
    debugprint("* Datasets loaded")

    for k, v in builder_fns.items():
        if v != False:
            args, kwargs = parse_args_kwargs(v)
            fn_map[k](builder, *args, **kwargs)
            debugprint(f"* Invoked '{fn_map[k].__name__}'")

    if mapbox_fig:
        args, kwargs = parse_args_kwargs(mapbox_fig)
        builder.set_mapbox(*args, **kwargs)
        debugprint("* Converted to MapBox figure")

    if adjust_focus:
        args, kwargs = parse_args_kwargs(adjust_focus)
        try:
            builder.adjust_focus(*args, **kwargs)
            debugprint("* Focus adjusted")
        except Exception:
            print("* Error when adjusting focus")

    if adjust_opacity:
        args, kwargs = parse_args_kwargs(adjust_opacity)
        try:
            builder.adjust_opacity(*args, **kwargs)
            debugprint("* Colorscale opacity adjusted")
        except Exception:
            print("* Error when adjusting colorscale opacity")

    builder.build_plot(**build_args)
    #builder.adjust_colorbar_size()

    if output_fig:
        args, kwargs = parse_args_kwargs(output_fig)
        builder.output_figure(*args, **kwargs)
        debugprint("* Figure output")

    if display_fig:
        args, kwargs = parse_args_kwargs(display_fig)
        builder.display_figure(*args, **kwargs)
        debugprint("* Figure displayed")

    return builder.get_plot_status()
