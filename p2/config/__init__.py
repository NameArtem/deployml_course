import os
ROOT = r'%s' % os.path.abspath(os.path.join(os.path.dirname( "src" ))).replace('\\','/')

with open(f"{ROOT}/VERSION", 'r') as version:
    __version__ = version.read().strip()