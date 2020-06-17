import datetime,pkg_resources

name = "miner3"
GIT_SHA = '$Id$'

try:
    __version__ = pkg_resources.get_distribution(name)
except:
    __version__ = 'development'

