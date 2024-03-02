#!/bin/bash

SOURCE_DIR="/Users/davidgao/Robotics/Repositories/universal_manipulation_interface"
TARGET_DIR="OrinNX:repositories/"
RSYNC_OPTIONS="-avzP"
## For Linux
# y,create,delete,move $SOURCE_DIR --format "%w%f" | while read file
## For MacOS
fswatch -o $SOURCE_DIR | while read change
do
  echo "Detected change in $file, syncing..."
  rsync $RSYNC_OPTIONS $SOURCE_DIR $TARGET_DIR --exclude-from='.gitignore';
done