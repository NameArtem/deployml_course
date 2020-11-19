import os


ROOT = r'%s' % os.path.abspath(os.path.join(os.path.dirname( "src" ), os.pardir, os.pardir)).replace('\\','/')
DATA_PATH = r'%s' % os.path.abspath(os.path.join(os.path.dirname( "src" ), os.pardir, os.pardir, 'data')).replace('\\','/')
SRC_ROOT = r'%s' % os.path.abspath(os.path.join(os.path.dirname( "src" ))).replace('\\','/')

with open(f"{SRC_ROOT}/VERSION", 'r') as version:
    __version__ = version.read().strip()