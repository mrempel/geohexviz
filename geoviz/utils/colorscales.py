from typing import Tuple, List, Union

import plotly.colors as cli
import math

ColorTuple = Tuple[int, ...]
ColorType = Tuple[int, ColorTuple]
ColorHelperType = List[ColorType]


def solid_scale(color: ColorTuple, min_scale: int = 0, max_scale: int = 1):
    """Retrieves a solid colorscale for a given color.

    :param color: The color to make a solid scale from
    :type color: ColorTuple
    :param min_scale: The minimum value on this colorscale
    :type min_scale: number
    :param max_scale: The maximum value on this colorscale
    :type max_scale: number
    :return: A solid colorscale
    :rtype: Tuple[Tuple[number, ColorTuple]]
    """
    return (min_scale, color), (max_scale, color)


def solidScales(colors: List[ColorTuple], min_scale: int = 0, max_scale: int = 1):
    """Retrieves a list of solid colorscales.

    :param colors: The list of colors to retrieve solid colorscales for
    :type colors: List[ColorTuple]
    :param min_scale: The minimum value for each solid scale
    :type min_scale: number
    :param max_scale: The maximum value for each solid scale
    :type max_scale: number
    :return: A list of solid colorscales
    :rtype: List[Tuple[number,ColorTuple]]
    """
    return [solid_scale(color, min_scale, max_scale) for color in colors]


def calculateLuminance(color: str) -> float:
    r, g, b = cli.unlabel_rgb(color)
    return math.sqrt(0.299 * r ** 2 + 0.587 * g ** 2 + 0.114 * b ** 2)


colormodules = {
    'sequential': cli.sequential,
    'diverging': cli.diverging,
    'qualitative': cli.qualitative,
    'cyclical': cli.qualitative
}


def getScale(scale_name, scale_type) -> list:
    if not scale_name:
        return cli.DEFAULT_PLOTLY_COLORS

    to_reverse = False
    if scale_name.endswith('_r'):
        scale_name = scale_name.replace('_r', '')
        to_reverse = True

    cs = getattr(colormodules[scale_type], scale_name)
    return list(reversed(cs)) if to_reverse else cs


def tryGetScale(scale_name):
    for k in colormodules:
        try:
            return getScale(scale_name, k)
        except AttributeError:
            pass
    raise AttributeError("No such color could be found.")


def alphaScale(scale: list, starting_alpha=0.3, decreasing: bool = False):
    if not decreasing:
        diff = 1 - starting_alpha
    else:
        diff = abs(0 - starting_alpha)

    increment = diff / len(scale)
    for i in range(len(scale)):
        scale[i] = configureColorWithAlpha(scale[i], starting_alpha + increment * i)


def configureColorWithAlpha(color, alpha: float = 0.4):
    if 'rgb' in color:
        return f'{color[:-1].replace("rgb", "rgba")}, {alpha})'
    else:
        return color


def configureScaleWithAlpha(scale, alpha: float = 0.4):
    unzipped = list(zip(*scale))
    try:
        sscale, colors = unzipped
    except ValueError:
        sscale, colors = None, unzipped

    if isinstance(scale[0], str):
        colors = scale

    colors, sscale = cli.convert_colors_to_same_type(colors, 'rgb', sscale)
    colors = [configureColorWithAlpha(pair, alpha=alpha) for pair in colors]
    return cli.make_colorscale(colors, scale=sscale)


def getDiscreteScale(scale_value: Union[str, list], scale_type: str, low: float, high: float, **kwargs):
    if isinstance(scale_value, str):
        scale_value = getScale(scale_value, scale_type)
    scale_value = cli.convert_colors_to_same_type(scale_value, 'rgb')[0]
    if scale_type == 'diverging':
        return discretize_diverging(scale_value, low, high, **kwargs)
    else:
        return discretize_sequential(scale_value, low, high, **kwargs)


def discretize(colors: List[str], size_portion: float = 0,
               center_portion: float = 0,
               fix_bound: bool = True,
               fix_extension: bool = True) -> List[Tuple[float, str]]:
    """Takes a single sequential colorscale and gets it's discrete form.

    :param colors: The list of colors on the scale
    :type colors: List[str]
    :param size_portion: The amount of space each discrete section will occupy on the scale (decimal-percentage)
    :type size_portion: float
    :param center_portion: The amount of space that the center will take up on the scale (decimal-percentage)
    :type center_portion: float
    :param fix_bound: Determines if a color is fixed if it goes over the top of the scale
    :type fix_bound: bool
    :param fix_extension: Determines if the last color should reach the end of the scale if no more colors available
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
    cv[0][0] = 0.0
    return cv


def discretize_sequential(colors: List[str], low: float, high: float, discrete_size: float = 1.0,
                          choose_hues: Union[List[int], int] = 1, choose_luminance: float = 0.0) -> \
        List[Tuple[float, str]]:
    """Makes a sequential scale discrete based on min, max on the scale.


    :param colors: The list of colors on the scale
    :type colors: List[str]
    :param low: The minimum numerical value on the scale (not percentage)
    :type low: float
    :param high: The maximum numerical value on the scale (not percentage)
    :type high: float
    :param discrete_size: The numerical amount that each discrete bar will occupy (not percentage)
    :type discrete_size: float
    :param choose_hues: Determines the step used in selecting colors from the scale, or the list of color positions that are used
    :type choose_hues: Union[List[int],int]
    :param choose_luminance: The maximum luminance of the colors to be selected
    :type choose_luminance: float
    :return: The discrete sequential scale
    :rtype: List[Tuple[float, str]]
    """

    colors = [color for color in colors if calculateLuminance(color) >= choose_luminance]
    try:
        colors = colors[::choose_hues]
    except TypeError:
        colors = [colors[i] for i in choose_hues]

    return discretize(colors, size_portion=discrete_size / (abs(low) + abs(high)))


def discretize_diverging(scale: List[str], low: float, high: float, discrete_size: float = 1.0,
                         remove_middle: bool = True, high_shading: bool = True, center: float = 0.0,
                         choose_left_hues: Union[List[int], int] = 1, choose_right_hues: Union[List[int], int] = 1,
                         choose_left_luminance: float = 0.0, choose_right_luminance: float = 0.0,
                         choose_luminance: float = 0.0) -> List[Tuple[float, str]]:
    """Transforms a diverging scale into a discrete diverging scale.

    :param scale: The list of colors on the scale
    :type scale: List[str]
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
    :return:
    :rtype:
    """
    total = abs(low) + abs(high)

    def getPercScale(num):
        return (num + abs(low)) / total

    left_hues = scale[:len(scale) // 2]
    right_hues = scale[(len(scale) // 2):]

    discrete_perc = discrete_size / total

    middle = None
    middle_left, middle_right = center, center

    if remove_middle:
        right_hues.pop(0)
    else:
        middle = right_hues.pop(0)
        middle_right, middle_left = center + discrete_size / 2, center - discrete_size / 2
        middle_left = middle_left if middle_left >= low else low
        middle_right = middle_right if middle_right <= high else high

    choose_left_luminance = choose_left_luminance if choose_left_luminance > choose_luminance else choose_luminance
    choose_right_luminance = choose_right_luminance if choose_right_luminance > choose_luminance else choose_luminance

    left_hues = [color for color in left_hues if choose_left_luminance <= calculateLuminance(color) >= choose_luminance]

    try:
        left_hues = left_hues[::choose_left_hues]
    except TypeError:
        left_hues = [left_hues[i] for i in choose_left_hues]

    right_hues = [color for color in right_hues if
                  choose_right_luminance <= calculateLuminance(color) >= choose_luminance]
    try:
        right_hues = right_hues[::choose_right_hues]
    except TypeError:
        right_hues = [right_hues[i] for i in choose_right_hues]

    percMiddleLeft, percMiddleRight = getPercScale(middle_left), getPercScale(middle_right)
    if high_shading:
        leftEdges = int(math.ceil(percMiddleLeft / discrete_perc))
        rightEdges = int(math.ceil(round(1 - percMiddleRight, 12) / round(discrete_perc, 12)))

        left_hues = left_hues[-leftEdges:]
        right_hues = right_hues[-rightEdges:]

    left = discretize(left_hues, size_portion=discrete_perc, fix_bound=False, fix_extension=False)
    right = discretize(right_hues, size_portion=discrete_perc, fix_bound=False, fix_extension=False)

    leftFactor = percMiddleLeft - left[-1][0]
    shiftedLeft = []
    for item in left:
        num = item[0] + leftFactor

        if num <= 0:
            continue
        else:
            shiftedLeft.append([num, item[1]])

    rightFactor = percMiddleRight - right[0][0]
    shiftedRight = []
    for item in right:
        num = item[0] + rightFactor
        if num >= 1:
            continue
        else:
            shiftedRight.append([num, item[1]])

    # print('Left', left)
    # print('Right', right)
    # print('\nShifted Left', shiftedLeft)
    # print('Shifted Right', shiftedRight)

    try:
        shiftedLeft.insert(0, [0, shiftedLeft[0][1]])
    except IndexError:
        pass
    try:
        shiftedRight[0][0] = shiftedLeft[-1][0] + 0.00001
    except IndexError:
        pass
    try:
        shiftedRight.append([1, shiftedRight[-1][1]])
    except IndexError:
        pass

    # ('Left', left)
    # print('Right', right)
    # print('\nShifted Left', shiftedLeft)
    # print('Shifted Right', shiftedRight)

    return [*shiftedLeft, *[[percMiddleLeft, middle], [percMiddleRight, middle]], *shiftedRight] if middle else [
        *shiftedLeft, *shiftedRight]


if __name__ == '__main__':
    x = getDiscreteScale('balance', 'diverging', [-6, -4, -3, -2, -1, 0, 1, 1.5], discrete_size=1, remove_middle=False,
                         center=0)
