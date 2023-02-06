#!/usr/bin/sh

echo "$(date +'%Y/%m/%d %T') - Start"
python -m app.executable.train_wordvector --n-limit=-1
echo "$(date +'%Y/%m/%d %T') - End"

