from typing import Tuple, List, Union

import plotly.colors as cli
import math

ColorTuple = Tuple[int, ...]
ColorType = Tuple[int, ColorTuple]
ColorHelperType = List[ColorType]


def solid_scale(color: str, min_scale: float = 0.0, max_scale: float = 1.0) -> Tuple:
    """Retrieves a solid colorscale for a given color.

    :param color: The color to make a solid colors from
    :type color: ColorTuple
    :param min_scale: The minimum value on this colorscale
    :type min_scale: number
    :param max_scale: The maximum value on this colorscale
    :type max_scale: number
    :return: A solid colorscale
    :rtype: Tuple[Tuple[number, str]]
    """
    return (min_scale, color), (max_scale, color)


def calculateLuminance(color: str) -> float:
    r, g, b = cli.unlabel_rgb(color)
    return math.sqrt(0.299 * r ** 2 + 0.587 * g ** 2 + 0.114 * b ** 2)


colormodules = {
    'sequential': cli.sequential,
    'diverging': cli.diverging,
    'qualitative': cli.qualitative,
    'cyclical': cli.qualitative
}


def get_scale_from_module(scale_name, scale_type) -> list:
    if not scale_name:
        return cli.DEFAULT_PLOTLY_COLORS

    to_reverse = False
    if scale_name.endswith('_r'):
        scale_name = scale_name.replace('_r', '')
        to_reverse = True

    cs = getattr(colormodules[scale_type], scale_name)
    return list(reversed(cs)) if to_reverse else cs


def get_scale(scale_name):
    for k in colormodules:
        try:
            return get_scale_from_module(scale_name, k)
        except AttributeError:
            pass
    raise AttributeError("No such colorscale could be found.")


def alphaScale(scale: list, starting_alpha=0.3, decreasing: bool = False):
    if not decreasing:
        diff = 1 - starting_alpha
    else:
        diff = abs(0 - starting_alpha)

    increment = diff / len(scale)
    for i in range(len(scale)):
        scale[i] = configure_color_opacity(scale[i], starting_alpha + increment * i)


def configure_color_opacity(color: str, alpha: float):
    if 'rgba' in color:
        r, g, b = cli.unlabel_rgb(color)
        return f'rgba({r},{g},{b},{alpha})'
    elif 'rgb' in color:
        return f'{color[:-1].replace("rgb", "rgba")}, {alpha})'
    else:
        return color


def configure_cscale_opacity(colorscale, alpha: float):
    try:
        colorscale = get_scale(colorscale)
    except AttributeError:
        pass

    f = get_cscale_format(colorscale)
    if f == 'nested-iterable':
        colors, scale = cli.convert_colors_to_same_type(cli.colorscale_to_colors(colorscale), 'rgb',
                                                        cli.colorscale_to_scale(colorscale))
        colors = [configure_color_opacity(color, alpha) for color in colors]
        return cli.make_colorscale(colors, scale)
    elif f == 'iterable':
        colorscale = cli.convert_colors_to_same_type(colorscale, 'rgb')[0]
        return [configure_color_opacity(color, alpha) for color in colorscale]
    elif f == 'dict':
        return {k: configure_color_opacity(v, alpha) for k, v in
                cli.convert_dict_colors_to_same_type(colorscale, colortype="rgb").items()}
    else:
        raise ValueError(
            "The colorscale is not of proper format for this function."
            " It must be in iterable, nested-iterable or map format."
        )


def configureScaleWithAlpha(scale, alpha: float = 0.4):
    unzipped = list(zip(*scale))
    try:
        sscale, colors = unzipped
    except ValueError:
        sscale, colors = None, unzipped

    if isinstance(scale[0], str):
        colors = scale

    colors, sscale = cli.convert_colors_to_same_type(colors, 'rgb', sscale)
    colors = [configure_color_opacity(pair, alpha=alpha) for pair in colors]
    return cli.make_colorscale(colors, scale=sscale)


def get_cscale_format(colorscale) -> str:
    """Retrieves the format of a colorscale.

    :param colorscale: The colorscale to get the format of
    :type colorscale: Any
    :return: The format of the colorscale (string, iterable, nested iterable, unknown)
    :rtype: str
    """
    if isinstance(colorscale, dict):
        return 'dict'
    else:
        if isinstance(colorscale, str):
            return 'string'
        try:
            if any(isinstance(i, list) or isinstance(i, tuple) or isinstance(i, set) for i in iter(colorscale)):
                return 'nested-iterable'
            else:
                return 'iterable'
        except TypeError:
            return 'unknown'


def discretize_cscale(colorscale, scale_type: str, low: float, high: float, **kwargs):
    """Transforms a normal colorscale into a discrete colorscale.
    
    :param colorscale: The colorscale to be converted
    :type colorscale: Any
    :param scale_type: The type of colorscale (sequential, diverging)
    :type scale_type: str
    :param low: The minimum value on the colors
    :type low: float
    :param high: The maximum value on the colors
    :type high: float
    :param kwargs: Keyword arguments for the discrete functions
    :type kwargs: **kwargs
    :return: The discretized colorscale
    :rtype: Any
    """
    if isinstance(colorscale, str):
        try:
            colorscale = get_scale(colorscale)
        except AttributeError:
            if get_cscale_format(colorscale) not in ['nested-iterable', 'iterable']:
                raise ValueError(
                    "The colors is in an invalid format. The colors must either be a named colorscale or a ")

    if any(isinstance(i, list) or isinstance(i, tuple) for i in iter(colorscale)):
        colorscale = cli.colorscale_to_colors(colorscale)

    colorscale = cli.convert_colors_to_same_type(colorscale, colortype="rgb", scale=None)[0]

    if scale_type == 'diverging':
        return discretize_diverging(colorscale, low, high, **kwargs)
    else:
        return discretize_sequential(colorscale, low, high, **kwargs)


def discretize(colors: List[str], size_portion: float = 0,
               center_portion: float = 0,
               fix_bound: bool = True,
               fix_extension: bool = True) -> List[Tuple[float, str]]:
    """Takes a single sequential colorscale and gets it's discrete form.

    :param colors: The list of colors on the colors
    :type colors: List[str]
    :param size_portion: The amount of space each discrete section will occupy on the colors (decimal-percentage)
    :type size_portion: float
    :param center_portion: The amount of space that the center will take up on the colors (decimal-percentage)
    :type center_portion: float
    :param fix_bound: Determines if a color is fixed if it goes over the top of the colors
    :type fix_bound: bool
    :param fix_extension: Determines if the last color should reach the end of the colors if no more colors available
    :type fix_extension: bool
    :return: The discrete colorscale
    :rtype: List[Tuple[float, str]]
    """
    cv = []
    color_ind = 0
    while True:
        try:
            prevPos = center_portion + size_portion * color_ind
            newPos = center_portion + size_portion * (color_ind + 1)
            if newPos >= 1:
                if fix_bound:
                    cv.extend([[prevPos + 0.000001, colors[color_ind]], [1.0, colors[color_ind]]])
                else:
                    cv.extend([[prevPos + 0.000001, colors[color_ind]], [newPos, colors[color_ind]]])
                break
            else:
                cv.extend([[prevPos + 0.000001, colors[color_ind]], [newPos, colors[color_ind]]])
            color_ind += 1
        except IndexError:
            if fix_extension:
                cv[len(cv) - 1][0] = 1.0
            break

    if not center_portion:
        cv[0][0] = 0.0
    return cv


def discretize_sequential(colors: List[str], low: float, high: float, discrete_size: float = 1.0,
                          choose_hues: Union[List[int], int] = 1, choose_luminance: float = 0.0) -> \
        List[Tuple[float, str]]:
    """Makes a sequential colors discrete based on min, max on the colors.

    choose_luminance does not work with colors that are not in rgb format.

    :param colors: The list of colors on the colors
    :type colors: List[str]
    :param low: The minimum numerical value on the colors (not percentage)
    :type low: float
    :param high: The maximum numerical value on the colors (not percentage)
    :type high: float
    :param discrete_size: The numerical amount that each discrete bar will occupy (not percentage)
    :type discrete_size: float
    :param choose_hues: Determines the step used in selecting colors from the colors, or the list of color positions that are used
    :type choose_hues: Union[List[int],int]
    :param choose_luminance: The maximum luminance of the colors to be selected
    :type choose_luminance: float
    :return: The discrete sequential colors
    :rtype: List[Tuple[float, str]]
    """

    if choose_luminance:
        colors = [color for color in colors if calculateLuminance(color) >= choose_luminance]
    try:
        colors = colors[::choose_hues]
    except TypeError:
        colors = [colors[i] for i in choose_hues]

    try:
        return discretize(colors, size_portion=discrete_size / (abs(low) + abs(high)))
    except IndexError:
        raise ValueError("There was most likely an issue with the amount of colors on the colors (not enough).")


def discretize_diverging(
        colors: List[str],
        low: float,
        high: float,
        discrete_size: float = 1.0,
        remove_middle: bool = True,
        high_shading: bool = True,
        center: float = None,
        center_hue: int = None,
        choose_left_hues: Union[List[int], int] = 1,
        choose_right_hues: Union[List[int], int] = 1,
        choose_left_luminance: float = 0.0,
        choose_right_luminance: float = 0.0,
        choose_luminance: float = 0.0
) -> List[Tuple[float, str]]:
    """Transforms a diverging scale into a discrete diverging scale.

    It should be noted that luminance parameters only work if the scale is in RGB format.
    THIS FEATURE IS HIGHLY EXPERIMENTAL.

    :param colors: The list of colors on the scale
    :type colors: List[str]
    :param low: The minimum value on the scale
    :type low: float
    :param high: The maximum value on the scale
    :type high: float
    :param discrete_size: The amount of space each discrete bin takes on the colorscale
    :type discrete_size: float
    :param remove_middle: Whether to remove the center of the diverging scale or not
    :type remove_middle: bool
    :param high_shading: Whether to use hues closer to the ends of the scale or not
    :type high_shading: bool
    :param center: The position of the center on the scale
    :type center: float
    :param center_hue: Where the center hue is in the list of colors given
    :type center_hue: int
    :param choose_left_hues: The list of color positions to use on the left of the scale or an integer skip for color selection on the left
    :type choose_left_hues: Union[List[int], int]
    :param choose_right_hues: The list of color positions to use on the right of the scale or an integer skip for color selection on the right
    :type choose_right_hues: Union[List[int], int]
    :param choose_left_luminance: Choose colors for the left side under the maximum luminance
    :type choose_left_luminance: float
    :param choose_right_luminance: Choose colors for the right side under the maximum luminance
    :type choose_right_luminance: float
    :param choose_luminance: Choose colors for the both sides of the scale under the maximum luminance
    :type choose_luminance: float
    :return: The discretized diverging colorscale
    :rtype: Any
    """
    total = abs(low) + abs(high)

    if center is None:
        center = (low + high) / 2

    if center_hue is None:
        center_hue = len(colors) // 2

    percentage_scale = lambda scale_position: (scale_position + abs(low)) / total

    left_hues = colors[:center_hue]
    right_hues = colors[center_hue:]

    discrete_perc = discrete_size / total

    middle = None
    middle_left, middle_right = center, center

    try:
        if remove_middle:
            right_hues.pop(0)
        else:
            middle = right_hues.pop(0)
            middle_right, middle_left = center + discrete_size / 2, center - discrete_size / 2
            middle_left = middle_left if middle_left >= low else low
            middle_right = middle_right if middle_right <= high else high
    except IndexError:
        raise ValueError("There was most likely an issue with the amount"
                         " of colors on the right of the colors (not enough).")

    choose_left_luminance = choose_left_luminance if choose_left_luminance > choose_luminance else choose_luminance
    choose_right_luminance = choose_right_luminance if choose_right_luminance > choose_luminance else choose_luminance

    if choose_luminance or choose_left_luminance:
        left_hues = [color for color in left_hues if
                     choose_left_luminance <= calculateLuminance(color) >= choose_luminance]

    try:
        left_hues = left_hues[::choose_left_hues]
    except TypeError:
        left_hues = [left_hues[i] for i in choose_left_hues]

    if choose_luminance or choose_left_luminance:
        right_hues = [color for color in right_hues if
                      choose_right_luminance <= calculateLuminance(color) >= choose_luminance]
    try:
        right_hues = right_hues[::choose_right_hues]
    except TypeError:
        right_hues = [right_hues[i] for i in choose_right_hues]

    percMiddleLeft, percMiddleRight = percentage_scale(middle_left), percentage_scale(middle_right)
    if high_shading:
        leftEdges = int(math.ceil(percMiddleLeft / discrete_perc))
        rightEdges = int(math.ceil(round(1 - percMiddleRight, 12) / round(discrete_perc, 12)))

        left_hues = left_hues[-leftEdges:]
        right_hues = right_hues[-rightEdges:]

    shiftedLeft = []
    try:
        left = discretize(left_hues, size_portion=discrete_perc, fix_bound=False, fix_extension=False)

        leftFactor = percMiddleLeft - left[-1][0]
        for item in left:
            num = item[0] + leftFactor

            if num <= 0:
                continue
            else:
                shiftedLeft.append([num, item[1]])
    except IndexError:
        pass

    shiftedRight = []
    try:
        right = discretize(right_hues, size_portion=discrete_perc, fix_bound=False, fix_extension=False)
        rightFactor = percMiddleRight - right[0][0]

        for item in right:
            num = item[0] + rightFactor
            if num >= 1:
                continue
            else:
                shiftedRight.append([num, item[1]])
    except IndexError:
        pass

    try:
        shiftedLeft.insert(0, [0, shiftedLeft[0][1]])
    except IndexError:
        pass
    try:
        shiftedRight[0][0] = percMiddleRight + 0.00001
    except IndexError:
        pass
    try:
        shiftedRight.append([1, shiftedRight[-1][1]])
    except IndexError:
        pass

    newlst = [*shiftedLeft, *[[percMiddleLeft, middle], [percMiddleRight, middle]], *shiftedRight] if middle else [
        *shiftedLeft, *shiftedRight]
    newlst[0][0] = 0.0
    newlst[-1][0] = 1.0

    return newlst


class _DBlock:

    def __init__(self, start: float, end: float, color: str, interp: float = 0.00001):
        self.start = start + interp
        self.end = end
        self.color = color

    def as_node(self, interp: float = 0.00001):
        return _DBlock(int(self.start) + interp, self.end, self.color)

    def as_list(self):
        return [self.start, self.color], [self.end, self.color]

    def onto_list(self, lst: list):
        lst.extend(self.as_list())

    def __copy__(self):
        return _DBlock(self.start, self.end, self.color, interp=0.0)

    def __str__(self):
        return f"Block\ns:\t\t{self.start}\ne:\t\t{self.end}\nc:\t\t{self.color}"


class _DScale:

    def __init__(self, start: float = 0.0, end: float = 1.0):
        self.start = start
        self.end = end
        self.blocks: List[_DBlock] = []

    def add_block(self, block: _DBlock):
        if block.end > self.end or block.start > self.end or block.start < self.start:
            raise ValueError("The block added is too large for the colors.")
        self.blocks.append(block)

    def end_block(self):
        try:
            return self.blocks[-1]
        except IndexError:
            raise IndexError("There is no end block, there are no blocks.")

    def start_block(self):
        try:
            return self.blocks[0]
        except IndexError:
            raise IndexError("There is no start block, there are no blocks.")

    def reverse_scale(self):
        revlst = list(reversed(self.blocks))
        lst = []
        for r, o in zip(revlst, self.blocks):
            lst.append(_DBlock(r.start, r.end, o.color, interp=0.0))
        self.blocks = lst

    def as_list(self):
        lst = []

        for block in self.blocks:
            block.onto_list(lst)
        return lst

    def __str__(self):
        joined = "\n".join(str(b.as_list()) for b in self.blocks)
        return f'DSCALE:\ns:\t{self.start}\ne:\t{self.end}\nBlocks:\n{joined}'


def discretize2(colors: List[str], size_portion: float = 0,
                center_portion: float = 0.0, size_low: float = 0.0, size_high: float = 1.0,
                fix_bound: bool = True,
                fix_extension: bool = True) -> _DScale:
    """Takes a single sequential colorscale and gets it's discrete form.

    :param colors: The list of colors on the colors
    :type colors: List[str]
    :param size_portion: The amount of space each discrete section will occupy on the colors (decimal-percentage)
    :type size_portion: float
    :param fix_bound: Determines if a color is fixed if it goes over the top of the colors
    :type fix_bound: bool
    :param fix_extension: Determines if the last color should reach the end of the colors if no more colors available
    :type fix_extension: bool
    :return: The discrete colorscale
    :rtype: List[Tuple[float, str]]
    """

    scale = _DScale(start=size_low, end=size_high)
    firstblock = _DBlock(size_low, size_low + (center_portion if center_portion else size_portion), colors.pop(0),
                         interp=0.0)
    scale.add_block(firstblock)

    for color_index, cv in enumerate(colors):

        block = _DBlock(firstblock.end + size_portion * color_index, firstblock.end + size_portion * (color_index + 1),
                        cv)
        if block.start >= scale.end:
            break
        if block.end >= scale.end and fix_bound:
            block.end = scale.end
            scale.add_block(block)
            break
        else:
            scale.add_block(block)

    scale.blocks[0].start = scale.start

    if fix_extension:
        scale.blocks[-1].end = scale.end

    return scale


from functools import reduce


def discretize_diverging2(scale: List[str], low: float, high: float, discrete_size: float = 1.0,
                          remove_middle: bool = True, center: float = None,
                          center_hue: int = None,
                          choose_left_hues: Union[List[int], int] = 1, choose_right_hues: Union[List[int], int] = 1,
                          choose_left_luminance: float = 0.0, choose_right_luminance: float = 0.0,
                          choose_luminance: float = 0.0) -> List[Tuple[float, str]]:
    total = abs(low) + abs(high)  # total space on colors
    perc_scale = lambda num: abs(num / total)
    if center is None:
        center = (low + high) / 2

    # split hues
    if center_hue is None:
        center_hue = len(scale) // 2

    left_hues = scale[:center_hue + 1]
    right_hues = scale[center_hue:]

    choose_left_luminance = choose_left_luminance if choose_left_luminance > choose_luminance else choose_luminance
    choose_right_luminance = choose_right_luminance if choose_right_luminance > choose_luminance else choose_luminance

    # an error may occur when choosing different luminance that removes the middle.
    if choose_luminance or choose_left_luminance:
        left_hues = [color for color in left_hues if
                     choose_left_luminance <= calculateLuminance(color) >= choose_luminance]

    try:
        left_hues = left_hues[::choose_left_hues]
    except TypeError:
        left_hues = [left_hues[i] for i in choose_left_hues]

    if choose_luminance or choose_left_luminance:
        right_hues = [color for color in right_hues if
                      choose_right_luminance <= calculateLuminance(color) >= choose_luminance]
    try:
        right_hues = right_hues[::choose_right_hues]
    except TypeError:
        right_hues = [right_hues[i] for i in choose_right_hues]

    if remove_middle:
        left_hues.pop()
        right_hues.pop(0)

    center_portion = perc_scale(discrete_size / 2) if not remove_middle else 0

    leftds = discretize(list(reversed(left_hues)), center_portion=center_portion,
                        size_portion=perc_scale(discrete_size), fix_bound=True, fix_extension=True,
                        size_low=perc_scale(center), size_high=1.0).as_list()
    rightds = discretize(right_hues, center_portion=center_portion, size_portion=perc_scale(discrete_size),
                         fix_bound=True, fix_extension=True, size_low=perc_scale(center), size_high=1.0).as_list()
    leftds = [(round(abs(i - 1), 6), v) for i, v in leftds]
    rightds = [(round(i, 6), v) for i, v in rightds]

    newlst = []
    if not remove_middle:
        leftds.pop(0)
        rightds.pop(0)
    newlst.extend(leftds)
    newlst.extend(rightds)
    newlst = list(sorted(newlst, key=lambda item: item[0]))

    return newlst
