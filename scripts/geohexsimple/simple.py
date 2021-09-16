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

    # TODO: in the future this should be substituted with another method of debug output
    if debug:
        debugprint = lambda x: print(x)
    else:
        debugprint = lambda x: None

    # print the interface
    base = os.path.basename(filepath)
    start = time.time()
    tok1 = "✨================ START ================✨"
    pathstr = f"Path: {filepath}"
    if len(pathstr) > len(tok1):
        m = (len(pathstr)-9)//2
        tok1 = f"✨{'='*m} START {'='*m}✨"

    m = (len(tok1)-12)//2
    tok2 = f"*{'-'*(m+1)} Plotting {'-'*(m+1)}*"
    tok3 = f"*{'-'*(len(tok1)-1)}*"
    m = (len(tok1) - 7) // 2
    tok4 = f"✨{'='*m} END {'='*m}✨"

    debugprint(tok1)
    debugprint(f"File: {base}")
    debugprint(f"Path: {filepath}")
    debugprint(tok2)

    # load and parse
    with open(filepath) as jse:
        read = json.load(jse)
    plot_adjustments = read.pop("plot_adjustments", {})
    output_fig = read.pop("output_figure", False)
    display_fig = read.pop("display_figure", True)
    data_adjustments = read.pop("data_adjustments", {})
    builder = PlotBuilder.builder_from_dict(**read)
    debugprint("* all data sets loaded")

    for k, v in data_adjustments.items():
        if v != False:
            args, kwargs = parse_args_kwargs(v)
            try:
                data_adjustments_map[k](builder, *args, **kwargs)
                debugprint(f"* invoked '{data_adjustments_map[k].__name__}'.")
            except Exception as e:
                try:
                    debugprint(f"* error while performing '{data_adjustments_map[k].__name__}' -> {e}")
                except KeyError:
                    debugprint(f"* no such data adjustment as: {k}")

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
                debugprint(f"* invoked '{plot_adjustments_map[k].__name__}'.")
            except Exception as e:
                try:
                    debugprint(f"* error while performing '{plot_adjustments_map[k].__name__}' -> {e}")
                except KeyError:
                    debugprint(f"* no such plot adjustment as: {k}")

    builder.finalize(raise_errors=False)

    if output_fig:
        args, kwargs = parse_args_kwargs(output_fig)
        builder.output(*args, **kwargs)
        debugprint("* figure output.")

    if display_fig:
        args, kwargs = parse_args_kwargs(display_fig)
        builder.display(*args, **kwargs)
        debugprint("* figure displayed.")

    end = time.time()
    debugprint(tok3)
    debugprint(f"Runtime: {round(end - start, 3)}s")
    debugprint(tok4)

    return builder.get_plot_status()


def get_json_filepath(message: str = ''):
    """Retrieves the path to a file from the user.
    """
    while True:
        filepath = input(message)
        if filepath.lower() == 'exit':
            sys.exit()
        elif filepath.lower() == 'back':
            return None
        elif filepath.endswith(".json"):
            return filepath
        print("The filepath must be a .JSON file.")


def get_json_directory(message: str = ''):
    """Retrieves a directory from the user.
    """
    while True:
        filepath = input(message)
        if filepath.lower() == 'exit':
            sys.exit()
        elif filepath.lower() == 'back':
            return None
        return filepath


def plot():
    """Plots a JSON file representing a plot.

    Asks the user for the required input.
    """
    filepath = get_json_filepath("Please input the location of your parameterized builder file (JSON).\n"
                                 "Options: json file path, back, exit.\n")
    if filepath is None:
        return
    run_json(filepath, debug=True)


def plotDir():
    """Plots a directory of JSON files representing separate plots.

    Asks the user for the required input.
    """
    directory = get_json_directory("Please input the location of a directory"
                                   " containing parameterized builder files (JSON).\n"
                                   "Options: json directory path, back, exit.\n")
    if directory is None:
        return
    try:
        for file in os.listdir(directory):
            if file.endswith('.json'):
                run_json(os.path.join(directory, file), debug=True)
    except NotADirectoryError:
        print('That was not a directory. Try again or exit.\n')


main_options = {
    'plot': plot,
    'plotdir': plotDir
}


def main():
    print("✨==================GeoSimple==================✨\n"
          " A script for the simple creation of\n"
          " hexagonally binned geospatial visualizations.\n"
          "✨=============================================✨")
    while (option_input := input("✨Main Menu✨\nOptions: plot, plotDir, exit\n").lower()) != 'exit':
        if option_input in main_options:
            main_options[option_input]()
        else:
            print('That was an incorrect option, try again.')
    sys.exit()


if __name__ == '__main__':
    main()
