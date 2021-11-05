__version__ = "1.0.0"

import os
import time
from typing import Dict, Callable
from geohexviz.builder import PlotBuilder, PlotStatus
from geohexviz.utils.util import parse_args_kwargs
import json
import yaml

_fn_map: Dict[str, Callable] = {
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

_ext_map = {
    ".yaml": "YAML",
    ".yml": "YAML",
    ".json": "JSON"
}


def run_yaml(filepath: str, strict: bool = False, debug: bool = False):
    """Runs a YAML representation of plot properties.

    :param filepath: The path to the YAML file
    :type filepath: str
    :param strict: Whether to handle strict events or not
    :type strict: bool
    :param debug: Whether to handle informational events or not
    :type debug: bool
    :return: The status of the final plot
    :rtype: PlotStatus
    """
    with open(filepath, "r", encoding='utf-8-sig') as stream:
        return _run_file_full(filepath, yaml.safe_load(stream), strict=strict, debug=debug)


def run_json(filepath: str, strict: bool = False, debug: bool = False):
    """Runs a JSON representation of plot properties.

    :param filepath: The path to the JSON file
    :type filepath: str
    :param strict: Whether to handle strict events or not
    :type strict: bool
    :param debug: Whether to handle informational events or not
    :type debug: bool
    :return: The status of the final plot
    :rtype: PlotStatus
    """
    with open(filepath) as jse:
        return _run_file_full(filepath, json.load(jse), strict=strict, debug=debug)


_ext_fn_map = {
    ".yaml": run_yaml,
    ".yml": run_yaml,
    ".json": run_json
}


def run_file(filepath: str, strict: bool = False, debug: bool = False):
    """Runs a file that represents plot properties.

    Accepted: YAML, JSON

    :param filepath: The filepath containing the plot properties
    :type filepath: str
    :param strict: Whether to handle strict events or not
    :type strict: bool
    :param debug: Whether to handle informational events or not
    :type debug: bool
    :return: The status of the final plot
    :rtype: PlotStatus
    """
    _, extension = os.path.splitext(filepath)
    try:
        return _ext_fn_map[extension](filepath, strict=strict, debug=debug)
    except KeyError:
        raise ValueError("There was an error running the file (invalid file extension).")


def _run_file_full(filepath: str, contents: dict, strict: bool = False, debug: bool = False):
    """Runs the contents of a file and provides output.

    :param filepath: The filepath for the plot properties
    :type filepath: str
    :param contents: The contents of the file in dict form
    :type contents: dict
    :param strict: Whether to handle strict events or not
    :type strict: bool
    :param debug: Whether to handle informational events or not
    :type debug: bool
    :return: The status of the final plot
    :rtype: PlotStatus
    """

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
    _, extension = os.path.splitext(filepath)
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
    try:
        debug_print(f"Type: {_ext_map[extension]}")
    except KeyError:
        debug_print(f"Type: {extension}")
    debug_print(tok2)

    status = _run_file_contents(contents, strict_fn, debug_print)

    end = time.time()
    debug_print(tok3)
    debug_print(f"Runtime: {round(end - start, 3)}s")
    debug_print(f"Plot Status: {status}")
    debug_print(tok4)

    return status


def _run_file_contents(contents: dict, strict_fn, debug_print):
    """Runs a json file representation of a plot scheme.

    :param contents: The contents of the file in dict form
    :type contents: dict
    :param strict_fn: The function that handles strict events
    :type strict_fn: object
    :param debug_print: The function that handles informational events
    :type debug_print: object
    :return: The plot status of the final plot
    :rtype: PlotStatus
    """

    default_fns_args = {
        "adjust_focus": {"on": "hexbin"},
        "adjust_positioning": True,
        "adjust_opacity": True
    }

    input_fns = contents.pop("functions", {})
    for k, v in default_fns_args.items():
        if k not in input_fns:
            input_fns[k] = v

    output = contents.pop("output", False)
    display = contents.pop("display", True)
    builder = PlotBuilder.builder_from_dict(**contents)
    debug_print("* all layers loaded")

    for k, v in input_fns.items():
        if v:
            args, kwargs = parse_args_kwargs(v)
            try:
                _fn_map[k](builder, *args, **kwargs)
                debug_print(f"* invoked function '{_fn_map[k].__name__}'.\nargs: {args}\nkwargs: {kwargs}")
            except Exception as e:
                try:
                    debug_print(f"* error while performing '{_fn_map[k].__name__}'.\nerror: {e}")
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

    return builder.get_plot_status()
