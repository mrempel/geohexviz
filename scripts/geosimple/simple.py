import os
import sys
import time
from typing import Dict, Callable
from geohexviz.builder import PlotBuilder
from geohexviz.utils.util import parse_args_kwargs
import json

fn_map: Dict[str, Callable] = {
    'remove_empties': PlotBuilder.remove_empties,
    'logify_scale': PlotBuilder.logify_scale,
    'generate_grid': PlotBuilder.auto_grid,
    'simple_clip': PlotBuilder.simple_clip,
    'clip_datasets': PlotBuilder.clip_datasets,
    'discretize_scale': PlotBuilder.discretize_scale,
}

data_adjustments_map: Dict[str, Callable] = {
    'remove_empties': PlotBuilder.remove_empties,
    'logify_scale': PlotBuilder.logify_scale,
    'generate_grid': PlotBuilder.auto_grid,
    'simple_clip': PlotBuilder.simple_clip,
    'clip_datasets': PlotBuilder.clip_datasets,
    'discretize_scale': PlotBuilder.discretize_scale,
}

plot_adjustments_map: Dict[str, Callable] = {
    'adjust_focus': PlotBuilder.adjust_focus,
    'adjust_opacity': PlotBuilder.adjust_opacity,
    'adjust_positioning': PlotBuilder.adjust_colorbar_size,
    'set_mapbox': PlotBuilder.set_mapbox
}


def run_json(filepath: str, debug: bool = False):
    """Runs a json file representation of a plot scheme.

    :param filepath: The filepath to the json file
    :type filepath: str
    :param debug: Determines if parts of the internal processes are printed or not
    :type debug: bool
    :return: The status of the plot after building
    :rtype: PlotStatus
    """

    if debug:
        debugprint = lambda x: print(x)
    else:
        debugprint = lambda x: None

    base = os.path.basename(filepath)
    start = time.time()
    debugprint("=============== START =================")
    debugprint(f"File: {base}")
    debugprint(f"Path: {filepath}")
    debugprint("-------------- Plotting ---------------")


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
    plot_adjustments = read.pop("plot_adjustments", {})
    # adjust_opacity = plot_adjustments.pop("opacity", True)
    # adjust_colorbar = plot_adjustments.pop("colorbar_size", True)
    # adjust_focus = plot_adjustments.pop("focus", True)
    # build_args.update(plot_adjustments.pop("build", {}))
    output_fig = read.pop("output_figure", False)
    display_fig = read.pop("display_figure", True)
    data_adjustments = read.pop("data_adjustments", {})

    builder = PlotBuilder.builder_from_dict(**read)
    debugprint("* Datasets loaded")

    for k, v in data_adjustments.items():
        if v != False:
            args, kwargs = parse_args_kwargs(v)
            try:
                data_adjustments_map[k](builder, *args, **kwargs)
                debugprint(f"* Invoked '{data_adjustments_map[k].__name__}'.")
            except Exception:
                try:
                    debugprint(f"* Error while performing '{plot_adjustments_map[k].__name__}'.")
                except KeyError:
                    debugprint(f"* No such data adjustment as: {k}")


    if 'adjust_focus' not in plot_adjustments:
        plot_adjustments['adjust_focus'] = True
    if 'adjust_opacity' not in plot_adjustments:
        plot_adjustments['adjust_opacity'] = True
    if 'adjust_positioning' not in plot_adjustments:
        plot_adjustments['adjust_positioning'] = True

    for k, v in plot_adjustments.items():
        if v != False:
            args, kwargs = parse_args_kwargs(v)
            try:
                plot_adjustments_map[k](builder, *args, **kwargs)
                debugprint(f"* Invoked '{plot_adjustments_map[k].__name__}'.")
            except Exception:
                try:
                    debugprint(f"* Error while performing '{plot_adjustments_map[k].__name__}'.")
                except KeyError:
                    debugprint(f"* No such plot adjustment as: {k}")

    '''
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
    '''


    # builder.build_plot(**build_args)
    builder.build_plot(raise_errors=False)

    if output_fig:
        args, kwargs = parse_args_kwargs(output_fig)
        builder.output_figure(*args, **kwargs)
        debugprint("* Figure output.")

    if display_fig:
        args, kwargs = parse_args_kwargs(display_fig)
        builder.display_figure(*args, **kwargs)
        debugprint("* Figure displayed.")

    end = time.time()
    debugprint("---------------------------------------")
    debugprint(f"Runtime: {round(end - start, 3)}s")
    debugprint("================ END ==================")

    return builder.get_plot_status()


def get_json_filepath(message: str = ''):
    while True:
        filepath = input(message)
        if filepath == 'exit':
            return None
        elif filepath.endswith(".json"):
            return filepath
        print("The filepath must be a .JSON file.")


def get_json_directory(message: str = ''):
    while True:
        filepath = input(message)
        return filepath if filepath != 'exit' else None


def plot():
    filepath = get_json_filepath("Please input the location of your parameterized builder file (JSON).")
    run_json(filepath, debug=True)


def plotDir():
    directory = get_json_directory("Please input the location of a directory"
                                   " containing parameterized builder files (JSON).")
    try:
        for file in os.listdir(directory):
            if file.endswith('.json'):
                run_json(os.path.join(directory, file), debug=True)
    except NotADirectoryError:
        print('That was not a directory. Try again or exit.')


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
