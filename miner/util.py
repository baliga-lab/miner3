import numpy, scipy, pandas, sklearn, lifelines, matplotlib, seaborn

import multiprocessing, multiprocessing.pool
import sys


def write_dependency_infos(outfile):
    py_version = map(str, list(sys.version_info[0:3]))
    outfile.write('Python %s\n\n' % ('.'.join(py_version)))
    outfile.write('Dependencies:\n')
    outfile.write('-------------\n\n')
    outfile.write('numpy: %s\n' % numpy.__version__)
    outfile.write('scipy: %s\n' % scipy.__version__)
    outfile.write('pandas: %s\n' % scipy.__version__)
    outfile.write('sklearn: %s\n' % sklearn.__version__)
    outfile.write('lifelines: %s\n' % lifelines.__version__)
    outfile.write('matplotlib: %s\n' % matplotlib.__version__)
    outfile.write('seaborn: %s\n' % seaborn.__version__)
