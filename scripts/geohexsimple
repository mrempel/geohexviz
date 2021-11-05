import os
from geohexviz.utils.file import run_file
import argparse


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
    if paths and any(i.endswith('.json') or i.endswith('.yaml') or i.endswith('.yml') for i in paths):
        for file in os.listdir(directory):
            _, ext = os.path.splitext(file)
            if ext in ['.json', '.yaml', '.yml']:
                run_file(os.path.join(directory, file), debug=debug, strict=strict)
    else:
        print("no json or yaml files found within the directory.")


def main():
    """Main executor.
    """
    my_parser = argparse.ArgumentParser(prog='geohexsimple', usage='%(prog)s [options]',
                                        description="Input plot property files to make hexagonally binned plots.")
    my_parser.add_argument('-p', '--path', type=str,
                           default="", dest='path', help='path to json file or directory containing '
                                                         'json files (required if no gui is used)')
    my_parser.add_argument('-g', '--gui', action='store_const', const=True,
                           default=False, dest='gui', help='enable command-line gui '
                                                           '(set to true if no path is provided)')
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

        while True:
            option_input = input("✨Main Menu✨\n"
                                 "Please input the location of your parameterized\nbuilder "
                                 "file (JSON, YAML) or a directory containing\nbuilder files.\n"
                                 "Options: file path, help, exit.\n")
            if option_input.lower() == 'exit':
                break
            if option_input.lower() == 'help':
                print("In order to use this script, a properly formatted JSON, or YAML file must be passed.\n"
                      "The user can also pass a directory of JSON files if they wish.\n")
            elif os.path.exists(option_input):
                try:
                    _plotDir(option_input, debug=not args.nofeedback, strict=args.verbose)
                except NotADirectoryError:
                    _, ext = os.path.splitext(option_input)
                    if ext in ['.json', '.yaml', '.yml']:
                        run_file(option_input, debug=not args.nofeedback, strict=args.verbose)
                    else:
                        print("The path you input exists, but is not a directory, JSON file, or YAML file.")
            else:
                print('That was an incorrect option or the file does not exist, try again.')
    else:
        if os.path.isdir(args.path):
            _plotDir(args.path, debug=not args.nofeedback, strict=args.verbose)
        else:
            run_file(args.path, debug=not args.nofeedback, strict=args.verbose)


if __name__ == '__main__':
    main()