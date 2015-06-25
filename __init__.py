# Read the version number from the VERSION file
from os.path import abspath, dirname, join
with open(join(dirname(abspath(__file__)), 'VERSION'), 'r') as version_file:
    __version__ = version_file.read().strip()

# Import internal files here so that user may skip the `lib`, eg. :
#     from deconv3d import Instrument
# instead of :
#     from deconv3d.lib import Instrument
from lib.instruments import *
from lib.run import *
from lib.spread_functions import *
from lib.masks import *
