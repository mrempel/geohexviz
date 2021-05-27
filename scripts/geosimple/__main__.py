from geoviz.builder import builder_from_dict
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
    read_param_file(fp)


def plotDir():
    while True:
        fp = get_file_input('Please input the location of '
                                      'your builder parameter files.')
        if not fp:
            return

        try:
            for file in os.listdir(fp):
                read_param_file(os.path.join(fp, file))
            break
        except NotADirectoryError:
            print('That was not a directory. Try again or exit.')


def read_param_file(filepath: str):
    with open(filepath) as jse:
        read = json.load(jse)

    builder = builder_from_dict(builder_dict=read)
    builder.build_plot()


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
