from util import run_json
import os
import sys
import time


def get_file_input(message: str = ''):
    while True:
        filepath = input(message)
        if filepath == 'exit':
            return None
        elif os.path.exists(filepath):
            return filepath

        print('That was not a valid filepath. Try again.')


def print_run(filepath: str):
    base = os.path.basename(filepath)
    start = time.time()
    print("=============== START =================")
    print(f"File: {base}")
    print(f"Path: {filepath}")
    print("-------------- Plotting ---------------")
    run_json(filepath, debug=True)
    end = time.time()
    print("---------------------------------------")
    print(f"Runtime: {round(end - start, 3)}s")
    print("================ END ==================")


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
    print_run(filepath)


def plotDir():
    directory = get_json_directory("Please input the location of a directory"
                                   " containing parameterized builder files (JSON).")
    try:
        for file in os.listdir(directory):
            if file.endswith('.json'):
                print_run(os.path.join(directory, file))
    except NotADirectoryError:
        print('That was not a directory. Try again or exit.')


def plotDir2():
    while True:
        fp = get_file_input('Please input the location of '
                            'your builder parameter files.')
        if not fp:
            return

        try:
            for file in os.listdir(fp):
                if file.endswith('.json'):
                    print_run(os.path.join(fp, file))
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
