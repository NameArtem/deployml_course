import os


ROOT = r'%s' % os.path.abspath(os.path.join(os.path.dirname( "src" ), os.pardir, os.pardir)).replace('\\','/')
DATA_PATH = r'%s' % os.path.abspath(os.path.join(os.path.dirname( "src" ), os.pardir, os.pardir, 'data')).replace('\\','/')