import os
import sys
import time
from typing import Dict, Callable
from geohexviz.builder import PlotBuilder
from geohexviz.utils.util import parse_args_kwargs
import json
import argparse

fn_map: Dict[str, Callable] = {
    'remove_empties': PlotBuilder.remove_empties,
    'logify_scale': PlotBuilder.logify_scale,
    'generate_grid': PlotBuilder.auto_grid,
    'simple_clip': PlotBuilder.simple_clip,
    'clip_layers': PlotBuilder.clip_layers,
    'discretize_scale': PlotBuilder.discretize_scale,
    'adjust_focus': PlotBuilder.adjust_focus,
    'adjust_opacity': PlotBuilder.adjust_opacity,
    'adjust_positioning': PlotBuilder.adjust_colorbar_size,
    'set_mapbox': PlotBuilder.set_mapbox
}


def run_json(filepath: str, strict: bool = False, debug: bool = False):
    """Runs a json file representation of a plot scheme.

    :param filepath: The filepath to the json file
    :type filepath: str
    :param debug: Determines if parts of the internal processes are printed or not
    :type debug: bool
    :param strict: Whether to print error messages during function applications or not
    :type strict: bool
    """

    default_fns_args = {
        "adjust_focus": {"on": "hexbin"},
        "adjust_positioning": True,
        "adjust_opacity": True
    }

    # TODO: in the future this should be substituted with another method of debug output
    if debug:
        debug_print = lambda x: print(x)
    else:
        debug_print = lambda x: None

    def strict_on(err: Exception):
        raise err

    if strict:
        strict_fn = strict_on
    else:
        strict_fn = lambda err: None

    # print the interface
    base = os.path.basename(filepath)
    start = time.time()
    tok1 = "✨================ START ================✨"
    pathstr = f"Path: {filepath}"
    if len(pathstr) > len(tok1):
        m = (len(pathstr) - 9) // 2
        tok1 = f"✨{'=' * m} START {'=' * m}✨"

    m = (len(tok1) - 12) // 2
    tok2 = f"*{'-' * (m + 1)} Plotting {'-' * (m + 1)}*"
    tok3 = f"*{'-' * (len(tok1) - 1)}*"
    m = (len(tok1) - 7) // 2
    tok4 = f"✨{'=' * m} END {'=' * m}✨"

    debug_print(tok1)
    debug_print(f"File: {base}")
    debug_print(f"Path: {filepath}")
    debug_print(tok2)

    # load and parse
    with open(filepath) as jse:
        read = json.load(jse)

    input_fns = read.pop("functions", {})
    for k, v in default_fns_args.items():
        if k not in input_fns:
            input_fns[k] = v

    output = read.pop("output", False)
    display = read.pop("display", True)
    builder = PlotBuilder.builder_from_dict(**read)
    debug_print("* all layers loaded")

    for k, v in input_fns.items():
        if v:
            args, kwargs = parse_args_kwargs(v)
            try:
                fn_map[k](builder, *args, **kwargs)
                debug_print(f"* invoked function '{fn_map[k].__name__}'.\nargs: {args}\nkwargs: {kwargs}")
            except Exception as e:
                try:
                    debug_print(f"* error while performing '{fn_map[k].__name__}'.\nerror: {e}")
                    strict_fn(e)
                except KeyError as f:
                    debug_print(f"* no such function as '{k}'.")
                    strict_fn(f)

    builder.finalize(raise_errors=False)

    if output:
        args, kwargs = parse_args_kwargs(output)
        builder.output(*args, **kwargs)
        debug_print("* figure output.")

    if display:
        args, kwargs = parse_args_kwargs(display)
        builder.display(*args, **kwargs)
        debug_print("* figure displayed.")

    end = time.time()
    debug_print(tok3)
    debug_print(f"Runtime: {round(end - start, 3)}s")
    debug_print(f"Plot Status: {builder.get_plot_status()}")
    debug_print(tok4)


def _plotDir(directory: str, debug: bool = False, strict: bool = False):
    """plots a directory of JSON files.

    :param directory: the path to the directory
    :type directory: str
    :param debug: whether to print messages during the run or not
    :type debug: bool
    :param strict: whether to raise errors during the run or not
    :type strict: bool
    """
    paths = os.listdir(directory)
    if paths and any(i.endswith('.json') for i in paths):
        for file in os.listdir(directory):
            if file.endswith('.json'):
                run_json(os.path.join(directory, file), debug=debug, strict=strict)
    else:
        print("no json files found within the directory.")


def main():
    my_parser = argparse.ArgumentParser(prog='simplecli', usage='%(prog)s [options]',
                                        description="Input JSON files to make hexagonally binned plots.")
    my_parser.add_argument('-p', '--path', type=str,
                           default="", dest='path', help='path to json file or directory containing json files')
    my_parser.add_argument('-g', '--gui', action='store_const', const=True,
                           default=False, dest='gui', help='enable command-line gui')
    my_parser.add_argument('-nf', '--nofeedback', action='store_const', const=True, default=False,
                           dest="nofeedback", help="turn off feedback while plotting")
    my_parser.add_argument('-v', '--verbose', action='store_const',
                           const=True, default=False, dest='verbose', help='whether to raise all errors or not')
    args = my_parser.parse_args()

    if not args.path:
        args.gui = True
    if args.gui:
        print("✨=================GeoHexSimple================✨\n"
              " A script for the simple creation of\n"
              " hexagonally binned geospatial visualizations.\n"
              "✨=============================================✨")

        while (option_input := input("✨Main Menu✨\n"
                                     "Please input the location of your parameterized\nbuilder "
                                     "file (JSON) or a directory containing\nbuilder files.\n"
                                     "Options: json file path, help, exit.\n").lower()) != 'exit':
            if option_input.lower() == 'help':
                print("In order to use this script, a properly formatted JSON file must be passed.\n"
                      "The user can also pass a directory of JSON files if they wish.\n")
            elif os.path.exists(option_input):
                try:
                    _plotDir(option_input, debug=not args.nofeedback, strict=args.verbose)
                except NotADirectoryError:
                    if option_input.endswith('.json'):
                        run_json(option_input, debug=not args.nofeedback, strict=args.verbose)
                    else:
                        print("The path you input exists, but is not a directory or json file.")
            else:
                print('That was an incorrect option or the file does not exist, try again.')
        sys.exit()
    else:
        if os.path.isdir(args.path):
            _plotDir(args.path, debug=not args.nofeedback, strict=args.verbose)
        else:
            run_json(args.path, debug=not args.nofeedback, strict=args.verbose)


if __name__ == '__main__':
    main()
