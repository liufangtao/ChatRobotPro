
import collections
import random  
import numpy as np
from tqdm import trange, tqdm
import h5py
from PIL import Image  
 

def resize_image(rgb_array,new_size=None):
    
    if new_size==None or new_size[:2]==rgb_array.shape[:2]:
        return rgb_array
    image = Image.fromarray(rgb_array.astype(np.uint8), 'RGB')  
    image = image.resize(new_size[:2][::-1])
    return np.array(image)

def _load_h5_data(file_path):

    data = []

    with h5py.File(file_path, 'r') as root:
        is_sim = root.attrs['sim']
        original_action_shape = root['/action'].shape
        episode_len = original_action_shape[0]
        #print(f"episode_len={episode_len}")

        camera_names = list(root[f'/observations/images/'].keys())
        if False:
            for cam_name in camera_names:
                shape = root[f'/observations/images/{cam_name}'][0].shape
                print(f"camera_names={cam_name},shape={shape}")
            
            print(f'action_shape={original_action_shape}')

        #episode_len=2
        for start_ts in trange(episode_len, leave=False):
        #for start_ts in range(episode_len):
        #for start_ts in trange(1):
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()

            for cam_name in camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]#.tolist()
            
            #for k in image_dict:
            #    image_dict[k]=resize_image(image_dict[k],self.IMAGE_SIZE[:2])
            # get all actions after and including start_ts
            
            #action = root['/action'][start_ts]
            
            action = root['/observations/qpos'][min(start_ts+2,episode_len-1)]
            step = {
                "images": image_dict, 
                "qpos":qpos,
                "qvel":qvel,
                "action":action,
                #'language_instruction':self.language_instruction
            }
            #yield step
            data.append(step )
        
        return data

def parse_episode_from_hdf5_v0(episode_info:dict):
    episode_path = episode_info['file']
    language_instructions = episode_info['language_instructions']
    image_name_map=episode_info.get("image_name_map",{})
    image_resize=episode_info.get("image_resize",None)
    data = _load_h5_data(episode_path)

    if not isinstance(language_instructions,set):
        if isinstance(language_instructions,str):
            language_instructions = {language_instructions}
        elif isinstance(language_instructions,collections.abc.Iterable):
            language_instructions = set(language_instructions)
        else:
            raise TypeError(language_instructions)


    episode = []
    for i, step in enumerate(data):

        images = {image_name_map.get(k,k):resize_image(v,image_resize) for k,v in step['images'].items()}
        instruction = (random.sample(language_instructions, 1)[0] 
                                      if language_instructions 
                                      else '')
        episode.append({
            'observation': {
                #**step["images"],
                **images,
                'state': step['qpos'],
            },
            'action': step['action'],
            'discount': 1.0,
            'reward': 0.0,
            'is_first': i == 0,
            'is_last': False,
            'is_terminal': False,
            'language_instruction':  instruction,
            #'language_embedding': language_embedding,
        })

    episode[-1]['reward']=1.0
    episode[-1]['is_last']=True
    episode[-1]['is_terminal']=True

    # create output data sample
    sample = {
        'steps': episode,
        'episode_metadata': {
            'file_path': episode_path
        }
    }
    #return sample

    return episode_path, sample

def parse_episode_from_hdf5(episode_info:dict):
    
    episode_path = episode_info['file']
    language_instructions = episode_info['language_instructions']
    observation_info=episode_info.get("observations",None)
    action_info=episode_info.get("actions",None)

    #action_shift = 
    #action_location = {a.name: a.location for a in action_info}
    state_location = {a.name: a.location for a in observation_info.states}   
    image_location = {img.name: img.location for img in observation_info.images.list}
    image_resize = observation_info.images.image_size


    if not isinstance(language_instructions,set):
        if isinstance(language_instructions,str):
            language_instructions = {language_instructions}
        elif isinstance(language_instructions,collections.abc.Iterable):
            language_instructions = set(language_instructions)
        else:
            raise TypeError(language_instructions)

    episode = []

    with h5py.File(episode_path, 'r') as root:

        episode_len = root[action_info[0].location].shape[0]

        if False:
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            #print(f"episode_len={episode_len}")
            camera_names = list(root[f'/observations/images/'].keys())
            for cam_name in camera_names:
                shape = root[f'/observations/images/{cam_name}'][0].shape
                print(f"camera_names={cam_name},shape={shape}")
            
            print(f'action_shape={original_action_shape}')

        instruction = (random.sample(language_instructions, 1)[0] 
                                      if language_instructions 
                                      else '')
        #episode_len=2
        #for start_ts in trange(episode_len, leave=False):
        for start_ts in range(episode_len):
            images = {k: resize_image(root[loc][start_ts], image_resize) for k,loc in image_location.items()}
            states = {k: root[loc][start_ts] for k,loc in state_location.items()}
            
            actions=dict()
            for action in action_info:
                loc = action.location
                if action.delta:
                    
                    shift = max(action.shift,1)
                    delta_mask=action.get("delta_mask", None) or [1]*action.dim
                    delta_mask = np.array(delta_mask).astype(bool)
                    shifted_action = root[loc][max(min(start_ts+shift,episode_len-1),0)]
                    current_action = root[loc][max(min(start_ts,episode_len-1),0)]
                    actions[action.name] = np.where(delta_mask, shifted_action - current_action, shifted_action) 
                else:
                    shifted_action = root[loc][max(min(start_ts+action.shift,episode_len-1),0)]
                    actions[action.name] = shifted_action
                
            episode.append({
                'observation': {
                    **images,
                    **states,
                },
                **actions,
                'discount': 1.0,
                'reward': 0.0,
                'is_first':start_ts == 0,
                'is_last': False,
                'is_terminal': False,
                'language_instruction':  instruction,
            })

        episode[-1]['reward']=1.0
        episode[-1]['is_last']=True
        episode[-1]['is_terminal']=True
        
    # create output data sample
    sample = {
        'steps': episode,
        'episode_metadata': {
            'file_path': episode_path
        }
    }
    #return sample
    return episode_path, sample



