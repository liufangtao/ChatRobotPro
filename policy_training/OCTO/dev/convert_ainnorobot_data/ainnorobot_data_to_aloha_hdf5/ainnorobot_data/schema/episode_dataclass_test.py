import numpy as np
from ainnorobot_data.schema.episode_dataclass import Episode, Step, Observation, Metadata

# Creating a dummy Episode object
dummy_episode = Episode(
    metadata=Metadata(
        dataset_name="test_dataset",
        episode_id=1,
        experiment_time="2024-07-15T10:00:00",
        operator="Test Operator",
        scene="Test Scene",
        environment="Test Environment",
        task_name="Test Task",
        task_name_candidates=["Test Task", "Sample Task"],
        goal_image=np.random.randint(0, 255, (64, 64, 3)).tolist(),
        goal_depth=np.random.randint(0, 255, (64, 64)).tolist(),
        sample_rate=30,
        num_steps=10,
        robot_name="Test Robot",
        robot_type="single_arm",
        robot_description="Test Description",
        robot_arm1_joints_state_dim=6,
        robot_arm2_joints_state_dim=6,
        robot_arm1_eef_state_dim=6,
        robot_arm2_eef_state_dim=6,
        robot_arm1_gripper_state_dim=3,
        robot_arm2_gripper_state_dim=3,
        robot_master_arm1_joints_state_dim=6,
        robot_master_arm2_joints_state_dim=6,
        robot_master_arm1_eef_state_dim=6,
        robot_master_arm2_eef_state_dim=6,
        robot_master_arm1_gripper_state_dim=3,
        robot_master_arm2_gripper_state_dim=3,
        robot_lift_state_dim=3,
        robot_base_state_dim=3,
        robot_arm1_joints_action_dim=6,
        robot_arm2_joints_action_dim=6,
        robot_arm1_eef_action_dim=6,
        robot_arm2_eef_action_dim=6,
        robot_arm1_gripper_action_dim=1,
        robot_arm2_gripper_action_dim=1,
        robot_lift_action_dim=3,
        robot_base_action_dim=3,
        camera1_rgb_resolution=[640, 480],
        camera2_rgb_resolution=[640, 480],
        camera1_depth_resolution=[640, 480],
        camera2_depth_resolution=[640, 480],
    ),
    steps=[
        Step(
            observation=Observation(
                lang_instruction="Move to the target",
                arm1_joints_state=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                arm2_joints_state=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                arm1_eef_state=[0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
                arm2_eef_state=[0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
                arm1_gripper_state=[0.1, 0.2, 0.3],
                arm2_gripper_state=[0.1, 0.2, 0.3],
                master_arm1_joints_state=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                master_arm2_joints_state=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                master_arm1_eef_state=[0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
                master_arm2_eef_state=[0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
                master_arm1_gripper_state=[0.1, 0.2, 0.3],
                master_arm2_gripper_state=[0.1, 0.2, 0.3],
                lift_state=[0.1, 0.2, 0.3],
                base_state=[0.1, 0.2, 0.3],
            ),
            arm1_joints_action=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            arm2_joints_action=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            arm1_eef_action=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            arm2_eef_action=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            arm1_gripper_action=[0.1],
            arm2_gripper_action=[0.1],
            lift_action=[0.1, 0.1, 0.1],
            base_action=[0.1, 0.1, 0.1],
            is_terminal=False,
            reward=1.0,
            discount=0.9
        ) for _ in range(2)
    ]
)

# Dumping to JSON
episode_json = dummy_episode.json(indent=4)

# Loading from JSON
new_episode = Episode.parse_raw(episode_json)

# Verifying the data
print(episode_json)
print(new_episode)
