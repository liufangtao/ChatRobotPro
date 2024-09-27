import random
import pickle as pkl
from pathlib import Path

import torchvision.transforms as transforms
from torch.utils.data import IterableDataset

class BCDataset(IterableDataset):
    def __init__(
        self,
        path,
        suite,
        scenes,
        tasks,
        obs_type,
        history,
        history_len,
        prompt,
        temporal_agg,
        num_queries,
        img_size,
        intermediate_goal_step=50,
        store_actions=False,
    ):
        self._obs_type = obs_type
        self._prompt = prompt
        self._history = history
        self._history_len = history_len if history else 1
        self.img_size = img_size
        self.intermediate_goal_step = intermediate_goal_step

        # temporal_aggregation
        self._temporal_agg = temporal_agg
        self._num_queries = num_queries

        # Convert task_names, which is a list, to a dictionary
        tasks = {task_name: scene[task_name] for scene in tasks for task_name in scene}

        # Get relevant task names
        task_name = []
        for scene in scenes:
            task_name.extend([task_name for task_name in tasks[scene]])

        # get data paths
        self._paths = []
        # for suite in suites:
        self._paths.extend(list((Path(path) / suite).glob("*")))

        if task_name is not None:
            paths = {}
            idx2name = {}
            for path in self._paths:
                task = str(path).split(".")[0].split("/")[-1]
                if task in task_name:
                    # get idx of task in task_name
                    idx = task_name.index(task)
                    paths[idx] = path
                    idx2name[idx] = task
            del self._paths
            self._paths = paths

        # store actions
        if store_actions:
            self.actions = []

        # read data
        self._episodes = {}
        self._all_episodes = []
        self._current_episode_idx = 0
        self._max_episode_len = 0
        self._max_state_dim = 0
        self._num_samples = 0
        for _path_idx in self._paths:
            print(f"Loading {str(self._paths[_path_idx])}")
            # read
            data = pkl.load(open(str(self._paths[_path_idx]), "rb"))
            observations = (
                data["observations"] if self._obs_type == "pixels" else data["states"]
            )
            actions = data["actions"]
            task_emb = data["task_emb"]
            self._episodes[_path_idx] = []
            for i in range(len(observations)):
                episode = dict(
                    observation=observations[i],
                    action=actions[i],
                    task_emb=task_emb,
                )
                self._episodes[_path_idx].append(episode)
                self._all_episodes.append((episode, _path_idx))
                self._max_episode_len = max(
                    self._max_episode_len,
                    (
                        len(observations[i])
                        if not isinstance(observations[i], dict)
                        else len(observations[i]["pixels"])
                    ),
                )
                # if obs_type == 'features':
                self._max_state_dim = max(
                    self._max_state_dim, data["states"][i].shape[-1]
                )
                self._num_samples += (
                    len(observations[i])
                    if self._obs_type == "features"
                    else len(observations[i]["pixels"])
                )

                # store actions
                if store_actions:
                    self.actions.append(actions[i])

        self.stats = {
            "actions": {
                "min": 0,
                "max": 1,
            },
            "proprioceptive": {
                "min": 0,
                "max": 1,
            },
        }
        self.preprocess = {
            "actions": lambda x: (x - self.stats["actions"]["min"])
            / (self.stats["actions"]["max"] - self.stats["actions"]["min"] + 1e-5),
            "proprioceptive": lambda x: (x - self.stats["proprioceptive"]["min"])
            / (
                self.stats["proprioceptive"]["max"]
                - self.stats["proprioceptive"]["min"]
                + 1e-5
            ),
        }

        # augmentation
        self.aug = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),])

    def _sample(self):
        episode, env_idx = self._all_episodes[self._current_episode_idx]
        self._current_episode_idx += 1
        # (Pdb++) episode['observation'].keys()
        # dict_keys(['robot_states', 'pixels', 'pixels_egocentric', 'joint_states', 'eef_states', 'gripper_states'])
        # (Pdb++) episode['observation']['robot_states'].shape
        # (122, 9)
        # (Pdb++) episode['observation']['pixels'].shape
        # (122, 128, 128, 3)
        # (Pdb++) episode.keys()
        # dict_keys(['observation', 'action', 'task_emb'])
        # (Pdb++) len(episodes['task_emb'])
        # 384
        # (Pdb++) episode['task_emb'].shape
        # (384,)
        # (Pdb++) episode['observation']['pixels_egocentric'].shape
        # (122, 128, 128, 3)
        # (Pdb++) episode['observation']['joint_states'].shape
        # (122, 7)
        # (Pdb++) episode['observation']['eef_states'].shape
        # (122, 7)
        # (Pdb++) episode['observation']['gripper_states'].shape
        # (122, 2)
        # (Pdb++) episode['action'].shape
        # (122, 7)
        # (Pdb++) episode['task_emb'].shape
        # (384,)


        observations = episode["observation"]
        actions = episode["action"]
        task_emb = episode["task_emb"]

        return (observations, actions, task_emb, str(self._paths[env_idx]))

    def __iter__(self):
        while self._current_episode_idx < len(self._all_episodes):
            yield self._sample()

    def __len__(self):
        return len(self._all_episodes)

def get_libero_dataset(path:str):
    suite="libero_90"
    scenes=['kitchen1', 'kitchen2', 'kitchen3', 'kitchen4', 'kitchen5', 'kitchen6', 'kitchen7', 'kitchen8', 'kitchen9', 'kitchen10', 'living1', 'living2', 'living3', 'living4', 'living5', 'living6', 'study1', 'study2', 'study3', 'study4']
    tasks = [{'kitchen1': 
              ['KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet', 
               'KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet', 
               'KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it', 
               'KITCHEN_SCENE1_put_the_black_bowl_on_the_plate', 
               'KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet']}, 
            {'kitchen2': 
             ['KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet', 
              'KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate', 
              'KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate', 
              'KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate', 
              'KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet', 
              'KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle', 
              'KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl']}, 
            {'kitchen3': 
             ['KITCHEN_SCENE3_put_the_frying_pan_on_the_stove', 
              'KITCHEN_SCENE3_put_the_moka_pot_on_the_stove', 
              'KITCHEN_SCENE3_turn_on_the_stove', 
              'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it']}, 
            {'kitchen4': 
             ['KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet', 
              'KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer', 
              'KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet', 
              'KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet', 
              'KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet', 
              'KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack']}, 
            {'kitchen5': 
             ['KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet', 
              'KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet', 
              'KITCHEN_SCENE5_put_the_black_bowl_on_the_plate', 
              'KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet', 
              'KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet']}, 
            {'kitchen6': 
             ['KITCHEN_SCENE6_close_the_microwave', 
              'KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug']}, 
            {'kitchen7': 
             ['KITCHEN_SCENE7_open_the_microwave', 
              'KITCHEN_SCENE7_put_the_white_bowl_on_the_plate', 
              'KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate']}, 
            {'kitchen8': 
             ['KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove', 
              'KITCHEN_SCENE8_turn_off_the_stove']}, 
            {'kitchen9': 
             ['KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf', 
              'KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet', 
              'KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf', 
              'KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet', 
              'KITCHEN_SCENE9_turn_on_the_stove', 
              'KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it']}, 
            {'kitchen10': 
             ['KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet', 
              'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it', 
              'KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet', 
              'KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it', 
              'KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it', 
              'KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it']}, 
            {'living1': 
             ['LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket', 
              'LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket', 
              'LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket', 
              'LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket']}, 
            {'living2': 
             ['LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket', 
              'LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket', 
              'LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket', 
              'LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket', 
              'LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket']}, 
            {'living3': 
             ['LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray', 
              'LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray', 
              'LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray', 
              'LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray', 
              'LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray']}, 
            {'living4': 
             ['LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray', 
              'LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray', 
              'LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray', 
              'LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray', 
              'LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray']}, 
            {'living5': 
             ['LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate', 
              'LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate', 
              'LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate', 
              'LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate']}, 
            {'living6': 
             ['LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate', 
              'LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate', 
              'LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate', 
              'LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate']}, 
            {'study1': 
             ['STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy', 
              'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy', 
              'STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy', 
              'STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy']}, 
            {'study2': 
             ['STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy', 
              'STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy', 
              'STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy', 
              'STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy']}, 
            {'study3': 
             ['STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy', 
              'STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy', 
              'STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy', 
              'STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy', 
              'STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy']}, 
            {'study4': 
             ['STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf', 
              'STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf', 
              'STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf', 
              'STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf']}]
    obs_type = 'pixels'
    history = False
    history_len = 10
    prompt = 'text'
    temporal_agg = True
    num_queries = 10
    img_size = 128
    intermediate_goal_step = 50
    store_actions = True
    dataset = BCDataset(
        path = path,
        suite=suite,
        scenes=scenes,
        tasks=tasks,
        obs_type=obs_type,
        history=history,
        history_len=history_len,
        prompt=prompt,
        temporal_agg=temporal_agg,
        num_queries=num_queries,
        img_size=img_size,
        intermediate_goal_step=intermediate_goal_step,
        store_actions=store_actions
    )
    return dataset


def get_libero_dataset_for_debugging(path:str):
    suite="libero_90"
    scenes=['kitchen1', 'living1']
    tasks = [{'kitchen1': 
              ['KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet', 
               'KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet']},   
             {'living1': 
              ['LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket']}, 
           ]
    obs_type = 'pixels'
    history = False
    history_len = 10
    prompt = 'text'
    temporal_agg = True
    num_queries = 10
    img_size = 128
    intermediate_goal_step = 50
    store_actions = True
    dataset = BCDataset(
        path = path,
        suite=suite,
        scenes=scenes,
        tasks=tasks,
        obs_type=obs_type,
        history=history,
        history_len=history_len,
        prompt=prompt,
        temporal_agg=temporal_agg,
        num_queries=num_queries,
        img_size=img_size,
        intermediate_goal_step=intermediate_goal_step,
        store_actions=store_actions
    )
    return dataset

if __name__ == '__main__':
    dataset = get_libero_dataset("/mnt/nas03/robot_mixed_raw_data/libero/")
    print(len(dataset))
