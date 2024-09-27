from pydantic import BaseModel
from typing import List


class Metadata(BaseModel):
    r"""
    描述：每个episode包含meta数据和steps数据，steps的每项数据为长度相等的数组.
    """
    dataset_name: str # 数据集名称 | 取值范围: 自定义后固定 | 说明: 与目录结构中的${datasetname}一致
    episode_id: int # 试验编号 | 取值范围: 0~2^31 | 说明: 每个episode对应1个完整的任务执行动作，与文件名中的${episodeID}一致
    experiment_time: str = None # 试验时间 | 取值范围: 自定义后固定 | 说明: 每次试验开始时刻的日期和时分秒，与文件名中的${experimenttime}一致
    operator: str = None # 操作人员名字 ｜ 取值范围: 自定义 | 说明: 每次试验的操作人员
    scene: str = None # 试验场景 | 取值范围: 自定义后固定 | 说明: 描述试验场景，与文件名中的$(scene)一致
    environment: str = None # 试验环境 ｜ 取值范围: 自定义后固定 | 说明: 描述机器人所在的具体环境，比如某个房间、某个厨房，与文件名中的${environment}一致
    task_name: str = None # 任务名称 | 取值范围: 自定义后固定 | 说明: 通常用于language instruction，与文件名中的${taskname}一致
    task_name_candidates: List[str] = None # 可能的任务名称 | 取值范围: 自定义后固定 | 说明: 同一个任务可能出现的不同表述方式, 存储为数组，自定义个数
    goal_image: List[List[List[int]]] = None # 目标图像 | 取值范围: 0~255 | 说明: 目标图像的像素数据，通常是完成任务之后的一帧RGB图像。shape=[H, W, C]
    goal_depth: List[List[int]] = None # 目标深度图像 | 取值范围: 0~2^31 | 说明: 目标深度图像的像素数据，通常是完成任务之后的一帧深度图像。shape=[H, W]
    sample_rate: int = None # 采样频率 ｜ 取值范围: 0~2^31 ｜ 说明: 每一秒钟采集多少个step，是相机帧率的采样率，所有其它数据通过上采样对齐到这个采样率
    num_steps: int # 总步数	 | 取值范围: 0~2^31 | 说明: 该任务包含的总steps
    robot_name: str = None # 机器人名称 | 取值范围: 自定义后固定 | 说明: 与文件名中的${robotname}一致
    robot_type: str = None # 机器人类型 | 取值范围: ["single_arm","dual_arm"] | 说明: 后续如有需要，可增加类型，比如人形机器人、机器狗等
    robot_description: str = None # 描述机器人的arm1、arm2分别指哪个机械臂, camera1、camera2分别指哪个摄像头，采集的state和action分别代表的内容和组织形式等
    robot_arm1_joints_state_dim: int = None # arm1机械臂关节状态参数量 | 取值范围: 0~255 | 说明: arm1机械臂的状态参数数量，跟关节数量一致。
    robot_arm2_joints_state_dim: int = None # arm2机械臂关节状态参数量 | 取值范围: 0~255 | 说明: arm2机械臂的状态参数数量，跟关节数量一致
    robot_arm1_eef_state_dim: int = None # arm1机械臂姿态参数量 | 取值范围: 6 | 说明: arm1机械臂姿态参数数量，按顺序包含：x/y/z和rx/ry/rz
    robot_arm2_eef_state_dim: int = None # arm2机械臂姿态参数量 | 取值范围: 6 | 说明: arm2机械臂姿态参数数量，按顺序包含：x/y/z和rx/ry/rz
    robot_arm1_gripper_state_dim: int = None # arm1夹爪状态参数量 | 取值范围: 0~255 | 说明: arm1夹爪总的状态参数数量，按顺序包含：位移、速度、力。未来如果采用灵巧手，关节再定义
    robot_arm2_gripper_state_dim: int = None # arm2夹爪状态参数量 | 取值范围: 0~255 | 说明: arm2夹爪总的状态参数数量，按顺序包含：位移、速度、力。未来如果采用灵巧手，关节再定义
    robot_master_arm1_joints_state_dim: int = None # master_arm1机械臂关节状态参数量 | 取值范围: 0~255 | 说明: 示教主动臂arm1机械臂的状态参数数量，跟关节数量一致，有时可作为action使用
    robot_master_arm2_joints_state_dim: int = None # master_arm2机械臂关节状态参数量 | 取值范围: 0~255 | 说明: 示教主动臂arm2机械臂的状态参数数量，跟关节数量一致，有时可作为action使用
    robot_master_arm1_eef_state_dim: int = None # master_arm1机械臂姿态参数量 | 取值范围: 6 | 说明: 示教主动臂arm1机械臂姿态参数数量，按顺序包含：x/y/z和rx/ry/rz，有时可作为action使用
    robot_master_arm2_eef_state_dim: int = None # master_arm2机械臂姿态参数量 | 取值范围: 6 | 说明: 示教主动臂arm2机械臂姿态参数数量，按顺序包含：x/y/z和rx/ry/rz，有时可作为action使用
    robot_master_arm1_gripper_state_dim: int = None # master_arm1夹爪状态参数量 | 取值范围: 0~255 | 说明: 示教主动臂arm1夹爪总的状态参数数量，按顺序包含：位移、速度、力。未来如果采用灵巧手，关节再定义，有时可作为action使用
    robot_master_arm2_gripper_state_dim: int = None # master_arm2夹爪状态参数量 | 取值范围: 0~255 | 说明: 示教主动臂arm2夹爪总的状态参数数量，按顺序包含：位移、速度、力。未来如果采用灵巧手，关节再定义，有时可作为action使用
    robot_lift_state_dim: int = None # 升降状态参数量 | 取值范围: 0~255 | 说明: 升降机构的所有状态参数数量，按顺序包含：升降机构位置
    robot_base_state_dim: int = None # 底盘状态参数量 | 取值范围: 0~255 | 说明: 底盘的所有状态参数数量，按顺序包含：底盘x/y坐标值、底盘旋转角度
    robot_arm1_joints_action_dim: int = None # arm1机械臂关节控制参数量 | 取值范围: 0～255 | 说明: arm1机械臂的关节控制参数数量。按顺序包含：1～n关节的旋转角度。机械臂根部为1号关节点。
    robot_arm2_joints_action_dim: int = None # arm2机械臂关节控制参数量 | 取值范围: 0～255 | 说明: arm2机械臂的关节控制参数数量。按顺序包含：1～n关节的旋转角度。机械臂根部为1号关节点。
    robot_arm1_eef_action_dim: int = None # arm1机械臂姿态控制参数量 | 取值范围: 6 | 说明: arm1机械臂姿态的控制参数数量，按照顺序包含：在基准坐标系上的Δx/Δy/Δz/Δrx/Δry/Δrz
    robot_arm2_eef_action_dim: int = None # arm2机械臂姿态控制参数量 | 取值范围: 6 | 说明: arm2机械臂姿态的控制参数数量，按照顺序包含：在基准坐标系上的Δx/Δy/Δz/Δrx/Δry/Δrz
    robot_arm1_gripper_action_dim: int = None # arm1夹爪控制参数量 | 取值范围: 0~255 | 说明: arm1夹爪的控制参数数量，状态值0或者1
    robot_arm2_gripper_action_dim: int = None # arm2夹爪控制参数量 | 取值范围: 0~255 | 说明: arm2夹爪的控制参数数量，状态值0或者1
    robot_lift_action_dim: int = None # 升降机构控制参数量 | 取值范围: 0-255 | 说明: 升降机构的控制参数数量，包含：升降机构位移（Δh）
    robot_base_action_dim: int = None # 底盘控制参数量  | 取值范围: 0-255 | 说明: 底盘的控制参数数量，按顺序包含：底盘前进或后退位移（Δs）、底盘旋转角度（Δθ）
    camera1_rgb_resolution: List[int] = None # camera1相机RGB图像分辨率 | 取值范围: 0～4096 | 说明: shape=[H, W]例如图像分辨率为4096×2160, 则可表示为[2160，4096], 数值可自行设定
    camera2_rgb_resolution: List[int] = None # camera2相机RGB图像分辨率 | 取值范围: 0～4096 | 说明: shape=[H, W]例如图像分辨率为4096×2160, 则可表示为[2160，4096], 数值可自行设定
    camera3_rgb_resolution: List[int] = None # camera3相机RGB图像分辨率 | 取值范围: 0～4096 | 说明: shape=[H, W]例如图像分辨率为4096×2160, 则可表示为[2160，4096], 数值可自行设定
    camera4_rgb_resolution: List[int] = None # camera4相机RGB图像分辨率 | 取值范围: 0～4096 | 说明: shape=[H, W]例如图像分辨率为4096×2160, 则可表示为[2160，4096], 数值可自行设定
    camera5_rgb_resolution: List[int] = None # camera5相机RGB图像分辨率 | 取值范围: 0～4096 | 说明: shape=[H, W]例如图像分辨率为4096×2160, 则可表示为[2160，4096], 数值可自行设定
    camera6_rgb_resolution: List[int] = None # camera6相机RGB图像分辨率 | 取值范围: 0～4096 | 说明: shape=[H, W]例如图像分辨率为4096×2160, 则可表示为[2160，4096], 数值可自行设定
    camera1_depth_resolution: List[int] = None # camera1相机depth图像分辨率 | 取值范围: 0～4096 | 说明: shape=[H, W]例如图像分辨率为4096×2160, 则可表示为[2160，4096], 数值可自行设定
    camera2_depth_resolution: List[int] = None # camera2相机depth图像分辨率 | 取值范围: 0～4096 | 说明: shape=[H, W]例如图像分辨率为4096×2160, 则可表示为[2160，4096], 数值可自行设定
    camera3_depth_resolution: List[int] = None # camera3相机depth图像分辨率 | 取值范围: 0～4096 | 说明: shape=[H, W]例如图像分辨率为4096×2160, 则可表示为[2160，4096], 数值可自行设定
    camera4_depth_resolution: List[int] = None # camera4相机depth图像分辨率 | 取值范围: 0～4096 | 说明: shape=[H, W]例如图像分辨率为4096×2160, 则可表示为[2160，4096], 数值可自行设定
    camera5_depth_resolution: List[int] = None # camera5相机depth图像分辨率 | 取值范围: 0～4096 | 说明: shape=[H, W]例如图像分辨率为4096×2160, 则可表示为[2160，4096], 数值可自行设定
    camera6_depth_resolution: List[int] = None # camera6相机depth图像分辨率 | 取值范围: 0～4096 | 说明: shape=[H, W]例如图像分辨率为4096×2160, 则可表示为[2160，4096], 数值可自行设定



class Observation(BaseModel):
    r"""an observation in a step"""
    lang_instruction: str = None #  实时的语言指令 ｜ 取值范围: 自定义 | 说明: shape=[]，更为细粒度的操作指令或纠正指令, 可以与meta中的task_name或者task_name_candidates搭配使用，一起作为prompt。
    arm1_joints_state: List[float] = None # arm1机械臂关节状态 | 取值范围: -2pi~2pi | 说明: shape=[arm1_joints_state_dim]。数据组织顺序：从机械臂根部到终端依次各关节点，单位为弧度。
    arm2_joints_state: List[float] = None # arm2机械臂关节状态 | 取值范围: -2pi~2pi | 说明: shape=[arm2_joints_state_dim]。数据组织顺序：从机械臂根部到终端依次各关节点，单位为弧度。
    arm1_eef_state: List[float] = None # arm1机械臂姿态 | 取值范围: 自定义 | 说明: shape=[6]。数据组织顺序: x/y/z/rx/ry/rz。xyz单位为米，rx/ry/rz单位为弧度。
    arm2_eef_state: List[float] = None # arm2机械臂姿态 | 取值范围: 自定义 | 说明: shape=[6]。数据组织顺序: x/y/z/rx/ry/rz。xyz单位为米，rx/ry/rz单位为弧度。
    arm1_gripper_state: List[float] = None # arm1夹爪状态 | 取值范围: 自定义 | 说明: shape=[arm1_gripper_state_dim]。数据组织顺序: 夹爪的位移、速度、力。位移单位为米，速度单位为米/秒，力单位为牛。未来如果采用灵巧手，再定义
    arm2_gripper_state: List[float] = None # arm2夹爪状态 | 取值范围: 自定义 | 说明: shape=[arm2_gripper_state_dim]。数据组织顺序: 夹爪的位移、速度、力。位移单位为米，速度单位为米/秒，力单位为牛。未来如果采用灵巧手，再定义
    master_arm1_joints_state: List[float] = None # master_arm1机械臂关节状态 | 取值范围: -2pi~2pi | 说明: shape=[master_arm1_joints_state_dim]。数据组织顺序：从机械臂根部到终端依次各关节点，单位为弧度。
    master_arm2_joints_state: List[float] = None # master_arm2机械臂关节状态 | 取值范围: -2pi~2pi | 说明: shape=[master_arm2_joints_state_dim]。数据组织顺序：从机械臂根部到终端依次各关节点，单位为弧度。
    master_arm1_eef_state: List[float] = None # master_arm1机械臂姿态 | 取值范围: 自定义 | 说明: shape=[6]。数据组织顺序: x/y/z/rx/ry/rz。xyz单位为米，rx/ry/rz单位为弧度。
    master_arm2_eef_state: List[float] = None # master_arm2机械臂姿态 | 取值范围: 自定义 | 说明: shape=[6]。数据组织顺序: x/y/z/rx/ry/rz。xyz单位为米，rx/ry/rz单位为弧度。
    master_arm1_gripper_state: List[float] = None # master_arm1夹爪状态 | 取值范围: 自定义 | 说明: shape=[master_arm1_gripper_state_dim]。数据组织顺序: 夹爪的位移、速度、力。位移单位为米，速度单位为米/秒，力单位为牛。未来如果采用灵巧手，再定义
    master_arm2_gripper_state: List[float] = None # master_arm2夹爪状态 | 取值范围: 自定义 | 说明: shape=[master_arm2_gripper_state_dim]。数据组织顺序: 夹爪的位移、速度、力。位移单位为米，速度单位为米/秒，力单位为牛。未来如果采用灵巧手，再定义
    lift_state: List[float] = None # 升降机构状态参数 | 取值范围: 自定义 | 说明: shape=[1]。数据组织顺序: 升降机构位置（h）。h单位为米。
    base_state: List[float] = None # 底盘姿态参数 | 取值范围: 自定义 | 说明: shape=[3]。数据组织顺序: 底盘位置坐标值（x/y）、底盘旋转角度（θ）。xy单位为米，θ单位为弧度。


class Step(BaseModel):
    r"""a step"""
    observation: Observation = None 
    arm1_joints_action: List[float] = None # arm1机械臂关节动作数据 | 取值范围: -2pi~2pi | 说明: shape=[arm1_joints_action_dim]。数据组织顺序: 1～n节点的旋转角度，单位为弧度。机械臂根部为1号关节点。
    arm2_joints_action: List[float] = None # arm2机械臂关节动作数据 | 取值范围: -2pi~2pi | 说明: shape=[arm2_joints_action_dim]。数据组织顺序: 1～n节点的旋转角度，单位为弧度。机械臂根部为1号关节点。
    arm1_eef_action: List[float] = None # arm1机械臂姿态动作数据 | 取值范围: 自定义，-2pi~2pi | 说明: shape=[6]。数据组织顺序: Δx/Δy/Δz/Δrx/Δry/Δrz。Δx/Δy/Δz的单位为米，Δrx/Δry/Δrz的单位为弧度。
    arm2_eef_action: List[float] = None # arm2机械臂姿态动作数据 | 取值范围: 自定义，-2pi~2pi | 说明: shape=[6]。数据组织顺序: Δx/Δy/Δz/Δrx/Δry/Δrz。Δx/Δy/Δz的单位为米，Δrx/Δry/Δrz的单位为弧度。
    arm1_gripper_action: List[float] = None # arm1夹爪动作数据 | 取值范围: 0/1 | 说明: shape=[1]。0代表关闭，1代表张开。
    arm2_gripper_action: List[float] = None # arm2夹爪动作数据 | 取值范围: 0/1 | 说明: shape=[1]。0代表关闭，1代表张开。
    lift_action: List[float] = None # 升降机构动作数据 | 取值范围: 自定义，-2pi~2pi | 说明: shape=[1]。数据组织顺序: 升降机构位移（Δh），Δh的单位为米。
    base_action: List[float] = None # 底盘动作数据 | 取值范围: 自定义，-2pi~2pi | 说明: shape=[2]。数据组织顺序: 底盘前进或后退位移（Δs）、底盘旋转角度（Δθ），Δs的单位为米，Δθ的单位为弧度。
    is_terminal: bool = None # 试验是否结束帧 | 取值范围: [True, False] | 说明: 如果所有的step中都不包含is_terminal为True的情况，说明当前episode并不完整
    reward: float = None # 奖励值 | 取值范围: [0, 1.0] | 说明: shape=[]，每一步的奖励值
    discount: float = None # 折扣值 | 取值范围: [0, 1.0] | 说明: shape=[]，每一步action执行的折扣值


class Episode(BaseModel):
    r"""an episode or a demo"""
    metadata: Metadata
    steps: List[Step]
