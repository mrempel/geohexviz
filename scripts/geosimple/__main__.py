from typing import Dict, Any, Callable
from geoviz.builder import PlotBuilder
from util import run_simple_JSON
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
