#!/bin/bash

echo "start inference program."

ws_path=$(pwd)
echo $ws_path

echo "start arm."
gnome-terminal --command="bash -c 'cd $ws_path/remote_control; sleep 2; echo 第二条命令; exec bash'"
