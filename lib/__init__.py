
from os.path import abspath, dirname

with open(dirname(abspath(__file__))+'/../VERSION', 'r') as version_file:
    __version__ = version_file.read().replace('\n', '').strip()

