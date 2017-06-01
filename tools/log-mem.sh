#!/bin/bash -e

LOGFILE=./mem-free.log
rm -f $LOGFILE

echo "      date     time $(free -m | grep total | sed -E 's/^    (.*)/\1/g')" >> $LOGFILE
while true; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') $(free -h | grep Mem: | sed 's/Mem://g')" >> $LOGFILE
    sleep 10
done
