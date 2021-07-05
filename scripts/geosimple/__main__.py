from typing import Dict, Any, Callable
from geoviz.builder import PlotBuilder
from utils import run_simple_JSON
import json
import os
import sys


def get_file_input(message: str = ''):
    while True:
        filepath = input(message)
        if filepath == 'exit':
            return None
        elif os.path.exists(filepath):
            return filepath

        print('That was not a valid filepath. Try again.')


def plot():
    fp = get_file_input('Please input the location of '
                        'your builder parameter file.')
    if not fp:
        return
    run_simple_JSON(fp)


def plotDir():
    while True:
        fp = get_file_input('Please input the location of '
                            'your builder parameter files.')
        if not fp:
            return

        try:
            for file in os.listdir(fp):
                run_simple_JSON(os.path.join(fp, file))
            break
        except NotADirectoryError:
            print('That was not a directory. Try again or exit.')


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


def fkey(d: dict):
    return next(iter(d))


def lkey(d: dict):
    return list(d.keys())[-1]


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
        print(args, kwargs)
        builder.adjust_opacity(*args, **kwargs)

    builder.build_plot(**build_args)

    if output_fig:
        args, kwargs = _parse_args_kwargs(output_fig)
        builder.output_figure(*args, **kwargs)

    if display_fig:
        args, kwargs = _parse_args_kwargs(display_fig)
        builder.display_figure(*args, **kwargs)


main_options = {
    'plot': plot,
    'plotdir': plotDir
}

if __name__ == '__main__':
    while (option_input := input('Select option.\nAvailable options: plot, plotDir, exit').lower()) != 'exit':
        if option_input in main_options:
            main_options[option_input]()
        else:
            print('That was an incorrect option, try again.')
    sys.exit()
