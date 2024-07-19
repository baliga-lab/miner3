import datetime
from importlib.metadata import version

name = "miner3"
GIT_SHA = '$Id$'

try:
    __version__ = version(name)
except:
    __version__ = 'development'

