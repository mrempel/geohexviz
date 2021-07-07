from typing import Dict, Any, Callable
from geoviz.builder import PlotBuilder
import json


def _parse_args_kwargs(item):
    if isinstance(item, dict):
        try:
            return item['args'], item['kwargs']
        except KeyError:
            return (), item
    elif isinstance(item, str):
        return [item], {}
    elif isinstance(item, list):
        return item, {}
    else:
        return (), {}


fn_map: Dict[str, Callable] = {
    'adjust opacity': PlotBuilder.adjust_opacity,
    'logarithmic scale': PlotBuilder.logify_scale,
    'generate grid': PlotBuilder.auto_grid,
    'adjust focus': PlotBuilder.adjust_focus,
    'simple clip': PlotBuilder.simple_clip,
    'clip datasets': PlotBuilder.clip_datasets,
    'discrete scale': PlotBuilder.discretize_scale,
}

output_fns = ['display figure', 'output figure']


def run_simple_JSON(filepath: str):
    with open(filepath) as jse:
        read = json.load(jse)

    build_args = {
        "raise_errors": False,
        "plot_regions": True,
        "plot_grids": True,
        "plot_outlines": True,
        "plot_points": True
    }

    mapbox_fig = read.pop("mapbox_token", False)
    adjustments = read.pop("adjustments", {})
    adjust_opacity = adjustments.pop("opacity", True)
    adjust_colorbar = adjustments.pop("colorbar_size", True)
    adjust_focus = adjustments.pop("focus", True)
    build_args.update(adjustments.pop("build", {}))
    output_fig = read.pop("output_figure", False)
    display_fig = read.pop("display_figure", True)
    builder_fns = read.pop("builder_functions", {})

    builder = PlotBuilder(**read)

    for k, v in builder_fns.items():
        if v != False:
            args, kwargs = _parse_args_kwargs(v)
            fn_map[k](builder, *args, **kwargs)

    if mapbox_fig:
        args, kwargs = _parse_args_kwargs(mapbox_fig)
        builder.set_mapbox(*args, **kwargs)

    if adjust_focus:
        args, kwargs = _parse_args_kwargs(adjust_focus)
        builder.adjust_focus(*args, **kwargs)

    if adjust_opacity:
        args, kwargs = _parse_args_kwargs(adjust_opacity)
        builder.adjust_opacity(*args, **kwargs)

    builder.build_plot(**build_args)

    if output_fig:
        args, kwargs = _parse_args_kwargs(output_fig)
        builder.output_figure(*args, **kwargs)

    if display_fig:
        args, kwargs = _parse_args_kwargs(display_fig)
        builder.display_figure(*args, **kwargs)

    return builder.get_plot_status()
