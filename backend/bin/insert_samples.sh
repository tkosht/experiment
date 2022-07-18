#!/usr/bin/sh

py_script="
from app.infra.localdb import LocalDb as DB

db = DB()
db.connect()
db.insert('Ev001', 'hello world')
db.close()
"

python -c "$py_script"

