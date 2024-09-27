ws_path=$(pwd)
echo $ws_path

echo "clean robot"
cd $ws_path/robot_ws
rm -rf build devel
cd ..

echo "clean agilex_ws"
cd $ws_path/agilex_ws
rm -rf build devel
cd ..

echo "clean camera"
cd $ws_path/camera_ws
rm -rf build devel
cd ..

echo "clean remote control"
cd $ws_path/remote_control
./tools/delete_build_file.sh
cd ..

echo "clean agilex"
cd $ws_path/agilex_ws
rm -rf build devel
cd ..