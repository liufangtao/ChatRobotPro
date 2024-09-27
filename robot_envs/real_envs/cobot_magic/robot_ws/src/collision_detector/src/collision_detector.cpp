#include <iostream>

#include <filesystem>

#include <ros/ros.h>

#include <urdf_parser/urdf_parser.h>

#include <moveit/robot_state/robot_state.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/utils/robot_model_test_utils.h>

#include <moveit/collision_detection/collision_common.h>

#include <moveit/collision_detection_fcl/collision_common.h>
#include <moveit/collision_detection_fcl/collision_env_fcl.h>

#include "moveit/move_group_interface/move_group_interface.h"

#include "collision_detector/CollisionState.h"

std::string str_mode = "collision";

collision_detection::AllowedCollisionMatrixPtr acm = nullptr;
collision_detection::CollisionEnvPtr c_env = nullptr;
moveit::core::RobotStatePtr ptr_robot_state;

// check self collision.
collision_detection::CollisionRequest col_req;
collision_detection::CollisionResult col_res;

std::vector<std::string> vec_left_joint_names = {
    "fl_joint1", "fl_joint2", "fl_joint3", "fl_joint4", "fl_joint5", "fl_joint6", "fl_joint7", "fl_joint8"
};
std::vector<std::string> vec_right_joint_names = {
    "fr_joint1", "fr_joint2", "fr_joint3", "fr_joint4", "fr_joint5", "fr_joint6", "fr_joint7", "fr_joint8"
};

void collision_on_sim(
    moveit::planning_interface::MoveGroupInterface& fl_move_group,
    moveit::planning_interface::MoveGroupInterface& fr_move_group)
{
    double fl_joint1 = -1.25, fl_joint2 = 1, fl_joint3 = 1.3, fl_joint4 = 0.0;
    double fr_joint1 = 1.05, fr_joint2 = 1, fr_joint3 = 1.3, fr_joint4 = 0.0;
    ros::Rate loop_rate(100);

    while(ros::ok()){
        if(fl_joint1 >= -1.50){  
            fl_joint1 += -0.05;
        }
        fl_joint2 += 0.05;
        fl_joint3 += 0.05;
        ptr_robot_state->setJointPositions("fl_joint1", &fl_joint1);
        ptr_robot_state->setJointPositions("fl_joint2", &fl_joint2);
        ptr_robot_state->setJointPositions("fl_joint3", &fl_joint3);
        if(fr_joint1 <= 1.5){
            fr_joint1 += 0.05;
        }
        fr_joint2 += 0.05;
        fr_joint3 += 0.05;
        ptr_robot_state->setJointPositions("fr_joint1", &fr_joint1);
        ptr_robot_state->setJointPositions("fr_joint2", &fr_joint2);
        ptr_robot_state->setJointPositions("fr_joint3", &fr_joint3);
        ptr_robot_state->update();

        ros::Time begin = ros::Time::now();
        fl_move_group.setJointValueTarget("fl_joint1", fl_joint1);
        fl_move_group.setJointValueTarget("fl_joint2", fl_joint2);
        fl_move_group.setJointValueTarget("fl_joint3", fl_joint3);
        moveit::core::MoveItErrorCode result = fl_move_group.move();
        fl_move_group.stop();
        ros::Duration d = ros::Time::now() - begin;

        fr_move_group.setJointValueTarget("fr_joint1", fr_joint1);
        fr_move_group.setJointValueTarget("fr_joint2", fr_joint2);
        fr_move_group.setJointValueTarget("fr_joint3", fr_joint3);
        result = fr_move_group.move();
        fr_move_group.stop();

        col_res.clear();
        c_env->checkSelfCollision(col_req, col_res, *ptr_robot_state, *acm);
        std::cout << "check self collision: " << col_res.collision << std::endl;
        std::cout << "check self distance: " << col_res.distance << std::endl;
        if(col_res.collision){
            std::cout << "ok\n";
        }

        loop_rate.sleep();
    }
}

bool collsion(
    collision_detector::CollisionStateRequest &req,
    collision_detector::CollisionStateResponse &res)
{
    std::string str_left_joints, str_right_joints;
    for(int i =0; i < req.left.size(); i++){
        str_left_joints += std::to_string(req.left[i]) + " ";
        str_right_joints += std::to_string(req.right[i]) + " ";
    }
    std::cout << "left joints: " << str_left_joints << std::endl;
    std::cout << "right joints: " << str_right_joints << std::endl;
    ROS_INFO("left joints: %s", str_left_joints.c_str());
    ROS_INFO("right joints: %s", str_right_joints.c_str());

    for(int i = 0; i < 6; i++){
        ptr_robot_state->setJointPositions(vec_left_joint_names[i], &req.left[i]);
        ptr_robot_state->setJointPositions(vec_right_joint_names[i], &req.right[i]);
    }
    ptr_robot_state->setJointPositions(vec_left_joint_names[6], &req.left[6]);
    ptr_robot_state->setJointPositions(vec_left_joint_names[7], &req.left[6]);
    ptr_robot_state->setJointPositions(vec_right_joint_names[6], &req.right[6]);
    ptr_robot_state->setJointPositions(vec_right_joint_names[7], &req.right[6]);
    ptr_robot_state->update(); 

    col_res.clear();
    c_env->checkSelfCollision(col_req, col_res, *ptr_robot_state, *acm);
    std::cout << "check self collision: " << col_res.collision << std::endl;
    std::cout << "check self distance: " << col_res.distance << std::endl;

    res.result = col_res.collision;

    return true;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "moveit_collision");

    std::cout << "argc: " << argc << "argv" << *argv << std::endl;
    if (argc >= 2) {
        str_mode = argv[1];
        std::cout << str_mode << std::endl;
    }

    ros::AsyncSpinner spinner(1);
    spinner.start();

    std::string str_path = std::filesystem::current_path().string();
    std::cout << str_path << std::endl;

    // load model file.
    const std::string str_urdf_path = str_path + "/src/mobile_aloha_sim/aloha_description/arx5_description/urdf/arx5_description_isaac.urdf";
    const std::string str_srdf_path = str_path + "/src/mobile_aloha_sim//arx5_moveit_config/config/arx5_description_copy.srdf";
    urdf::ModelInterfaceSharedPtr urdf_model = urdf::parseURDFFile(str_urdf_path);
    if (urdf_model == nullptr)
    {
        std::cout << "load urdr failed\n";
    }

    srdf::ModelSharedPtr srdf_model = std::make_shared<srdf::Model>();
    srdf_model->initFile(*urdf_model, str_srdf_path);

    moveit::core::RobotModelPtr robot_model = std::make_shared<moveit::core::RobotModel>(urdf_model, srdf_model);
    if(robot_model == nullptr){
        std::cout << "load robot model failed\n";
    }

    acm = std::make_shared<collision_detection::AllowedCollisionMatrix>(*robot_model->getSRDF());

    c_env = std::make_shared<collision_detection::CollisionEnvFCL>(robot_model);

    ptr_robot_state = std::make_shared<moveit::core::RobotState>(robot_model);

    col_req.distance = true;
    col_req.contacts = true;
    col_req.max_contacts = 10;
    col_req.verbose = true;

    c_env->checkSelfCollision(col_req, col_res, *ptr_robot_state, *acm);
    std::cout << "check self collision: " << col_res.collision << std::endl;
    ROS_INFO("Check self collision: %d", col_res.collision);

    
    if(str_mode == "collision_on_sim"){
        std::string str_fl_arm = "fl_arm";
        std::string str_fr_arm = "fr_arm";

        moveit::planning_interface::MoveGroupInterface fl_move_group(str_fl_arm);
        moveit::planning_interface::MoveGroupInterface fr_move_group(str_fr_arm);

        collision_on_sim(fl_move_group, fr_move_group);
    }
    else if (str_mode == "collision") {
        ros::MultiThreadedSpinner spinner(1);
        ros::NodeHandle n;
        ros::ServiceServer collsion_service = n.advertiseService("/collision_state", collsion);

        ROS_INFO("Ready to collision detector.");
        spinner.spin();
    }

    return 0;
}
