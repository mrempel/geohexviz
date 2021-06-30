from typing import Dict, Any, Callable
from geoviz.builder import PlotBuilder
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
    execute_simple_file(fp)


def plotDir():
    while True:
        fp = get_file_input('Please input the location of '
                            'your builder parameter files.')
        if not fp:
            return

        try:
            for file in os.listdir(fp):
                execute_simple_file(os.path.join(fp, file))
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
    'set mapbox': PlotBuilder.set_mapbox,
    'output figure': PlotBuilder.output_figure,
    'display figure': PlotBuilder.display_figure,
    'clear figure': PlotBuilder.clear_figure,
    'update main manager': PlotBuilder.update_main_manager,
    'build plot': PlotBuilder.build_plot
}

fn_map2 = {' '.join(func.split('_')): getattr(PlotBuilder, func) for func in dir(PlotBuilder) if
           callable(getattr(PlotBuilder, func)) and not func.startswith('_')}
print(fn_map2)

output_fns = ['display figure', 'output figure']


# TODO: this seems to be the easy option
# TODO: it limits the users ability to input things in a certain order
def execute_simple_file(filepath: str):
    with open(filepath) as jse:
        read = json.load(jse)

    default_fns = {
        'clip datasets': False,
        'simple clip': False,
        'generate grid': False,
        'set mapbox': False,
        'logarithmic scale': False,
        'discrete scale': False,
        'adjust opacity': True,
        'adjust focus': True,
        'output figure': False,
        'display figure': {'renderer': 'browser'}
    }

    for k, v in read.pop("plot functions", {}).items():
        if k not in default_fns:
            raise ValueError(f"Received an invalid function. Valid functions: {list(default_fns.keys())}.")
        default_fns[k] = v

    opf = default_fns.pop('output figure')
    dpf = default_fns.pop('display figure')
    default_fns['build plot'] = {'raise_errors': False}
    default_fns['output figure'] = opf
    default_fns['display figure'] = dpf

    builder = PlotBuilder(**read)

    for k, v in default_fns.items():
        if v != False:
            args, kwargs = _parse_args_kwargs(v)
            fn_map2[k](builder, *args, **kwargs)


def execute_file(filepath: str):
    with open(filepath) as jse:
        read = json.load(jse)

    builder_fns = read.pop('builder_functions', [])
    if builder_fns:
        if fkey(builder_fns[-1]) not in output_fns:
            builder_fns.append({'display figure': True})
    else:
        builder_fns.append({'display figure': True})

    builder = PlotBuilder(**read)

    for k in builder_fns:
        v = k[fkey(k)]
        k = fkey(k)
        if v != False:
            args, kwargs = _parse_args_kwargs(v)
            try:
                print(k, args, kwargs)
                fn_map2[k](builder, *args, **kwargs)
            except KeyError:
                raise ValueError(f"There was a function that was not expected. Received {k}.")


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
