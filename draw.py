# お絵描きライブラリ


def tochar(f):
    if f < 0.2:
        return '.'
    elif f < 0.5:
        return '+'
    elif f < 0.8:
        return '#'
    else:
        return '@'


def mnist(x):
    """
    28x28 のお絵描き
    """
    height = 28
    for i in range(height):
        print(' '.join(map(tochar, x[i * height:i * height + height])))


def seq(x):
    print(''.join(map(tochar, x)))
