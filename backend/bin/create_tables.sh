#!/usr/bin/sh

py_script="
from app.infra.localdb import LocalDb as DB

db = DB()
db.connect()
db.create_tables()
db.close()
"

python -c "$py_script"

