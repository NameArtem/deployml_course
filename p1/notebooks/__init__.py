import os


ROOT = r'%s' % os.path.abspath(os.path.join(os.path.dirname( "notebooks" ), os.pardir)).replace('\\','/')
DATA_PATH = r'%s' % os.path.abspath(os.path.join(os.path.dirname( "notebooks" ), os.pardir, 'data')).replace('\\','/')