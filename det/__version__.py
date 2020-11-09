__version__ = '1.0.0'


def parse_version_info(version_str):
    version_info = []
    for x in version_str.split('.'):
        version_info.append(int(x))
    return tuple(version_info)


version_info = parse_version_info(__version__)