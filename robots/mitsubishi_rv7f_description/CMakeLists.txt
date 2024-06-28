cmake_minimum_required(VERSION 2.8.3)
project(mitsubishi_rv7f_description)
find_package(catkin REQUIRED)

find_package(roslaunch)
roslaunch_add_file_check(test/rv7f/rv7f_launch_test.xml)
