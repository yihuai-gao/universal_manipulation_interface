#!/bin/bash
SCRIPT_DIR=$(realpath $(dirname $0))
WORKSPACE_DIR=$(realpath $SCRIPT_DIR/..)

TARGET_DIR="OrinNX:repositories/"
RSYNC_OPTIONS="-avzP"
SYSTEM=$(uname -s)
if [ $SYSTEM = "Linux" ]; then
  ## For Linux
  inotifywait -m -r -e modify,create,delete,move $WORKSPACE_DIR --format "%w%f" | while read file
do
  echo "Detected change in $file, syncing..."
  rsync $RSYNC_OPTIONS $WORKSPACE_DIR $TARGET_DIR --exclude-from='.gitignore';
done

elif [ $SYSTEM = "Darwin" ]; then
  ## For MacOS
  fswatch -o $WORKSPACE_DIR | while read change
do
  echo "Detected change in $file, syncing..."
  rsync $RSYNC_OPTIONS $WORKSPACE_DIR $TARGET_DIR --exclude-from='.gitignore';
done
fi