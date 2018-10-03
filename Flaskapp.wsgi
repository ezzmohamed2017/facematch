#!/usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/FlaskApp/")
sys.path.append("/restAPI/lib/python3.5/site-packages")
sys.path.append("/restAPI/bin/")

from FlaskApp import app as application
application.secret_key ='1234567'
