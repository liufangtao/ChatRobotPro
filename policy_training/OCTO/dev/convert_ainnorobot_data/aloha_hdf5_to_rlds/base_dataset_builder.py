import logging
import time
import traceback
from typing import Iterator, List, Tuple, Any, Type

import glob
import numpy as np
import tensorflow_datasets as tfds
from tqdm import trange, tqdm
import os,sys
from PIL import Image  
import data_utils
import itertools


DEFULT_IMAGE_NAME_MAP={"cam_high": "top",
                #"cam_low": "low",
                "cam_left_wrist":"left_wrist",
                "cam_right_wrist":"right_wrist",}

default_language_instruction='Grab the coke with your left arm and put it into the plate'


class BaseDatasetBuilder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for test dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        
        #hdf5_path=kwargs.pop("hdf5_path")
        assert "source" in kwargs

        
        self.source = kwargs.pop("source")
        #self.image_info = kwargs.pop("image_info")
        #self.hdf5_path = kwargs.pop("hdf5_path")
        #image_size = kwargs.pop("image_size",None) or (480, 640)
        #self.image_encoding = kwargs.pop("image_encoding",'jpeg')
        #self.image_size = (*image_size,3)
        self.max_workers = kwargs.pop("max_workers",1)
        #self.image_map = kwargs.pop("image_map",DEFULT_IMAGE_NAME_MAP)
        #self.image_location = {img.name:img.location for img in self.image_info.images}
        self.observatoins = kwargs.pop("observatoin_info")
        self.actions = kwargs.pop("actions_info")
        #self.language_instruction = kwargs.pop("language_instruction",default_language_instruction)
        #self.hdf5_path = kwargs.get("hdf5_path")
        #self.hdf5_path = '/mnt/nas03/robotdatas/data20240717/aloha_mobile_dummy/'
        #self.language_instruction = 'Grab the coke with your left arm and put it into the plate'
        
        #self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
        super().__init__(*args, **kwargs)
        
    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        **{img.name: tfds.features.Image(
                            shape=self.observatoins.images.image_size,
                            dtype=np.uint8,
                            encoding_format=self.observatoins.images.image_encoding,
                            doc=img.doc,
                        ) for img in self.observatoins.images.list},
                        #'top': tfds.features.Image(
                        #    shape=self.image_size,
                        #    dtype=np.uint8,
                        #    encoding_format=self.image_encoding,
                        #    doc='Main camera RGB observation.',
                        #),
                        #'left_wrist': tfds.features.Image(
                        #    shape=self.image_size,
                        #    dtype=np.uint8,
                        #    encoding_format=self.image_encoding,
                        #    doc='Wrist camera RGB observation.',
                        #),
                        #'right_wrist': tfds.features.Image(
                        #    shape=self.image_size,
                        #    dtype=np.uint8,
                        #    encoding_format=self.image_encoding,
                        #    doc='Wrist camera RGB observation.',
                        #),
                        **{s.name: tfds.features.Tensor(
                            shape=(s.dim,),
                            dtype=np.float32,
                            doc=s.doc,
                        ) for s in self.observatoins.states},
                        #'state': tfds.features.Tensor(
                        #    shape=(14,),
                        #    dtype=np.float32,
                        #    doc='Robot joint pos (two arms + grippers).',
                        #)
                    }),
                    **{
                        a.name: tfds.features.Tensor(
                        shape=(a.dim,),
                        dtype=np.float32,
                        doc=a.doc,
                    ) for a in self.actions},
                    
                    #'action': tfds.features.Tensor(
                    #    shape=(14,),
                    #    dtype=np.float32,
                    #    doc='Robot action for joints in two arms + grippers.',
                    #),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    #'language_embedding': tfds.features.Tensor(
                    #    shape=(512,),
                    #    dtype=np.float32,
                    #    doc='Kona language embedding. '
                    #        'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    #),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(),
            #'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _generate_examples(self) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        # create list of all examples
        hdf5_sources = []
        tfds_sources = []
        for s in self.source:
            data_type = s.get("type","hdf5")
            if data_type == "hdf5":
                path,language_instruction = s.hdf5_path, s.language_instruction
                if isinstance(path,str):
                    path = [path]
                episode_paths = []
                for p in path:
                    assert os.path.exists(p),f"路径不存在：{p}"
                    files = glob.glob(os.path.join(p,"*.hdf5"))
                    assert len(files)>0,f"没找到HDF5文件 : {p}"
                    episode_paths.extend(files)
                max_example_num=s.get("max_example_num",None)
                if max_example_num:
                    episode_paths = episode_paths[:max_example_num]

                if len(episode_paths)==0:
                    logging.warn(f"没找到HDF5文件 : {s.hdf5_path}")

                #num_episodes = len(episode_paths)
                for p in episode_paths:
                    hdf5_sources.append( dict(file=p,
                                    language_instructions={language_instruction},
                                    #image_location=self.image_location,                      
                                    #image_resize=self.image_size[:2]
                                    observations=self.observatoins,
                                    actions=self.actions,
                                    ))
            elif data_type == "tfds":
                path = os.path.join(s.directory,s.name,s.version)
                assert os.path.exists(path),f"路径不存在：{path}"
                tfds_sources.append( dict(directory=s.directory,
                                            name=s.name, 
                                            version=s.version,
                                            max_example_num=s.get("max_example_num",None)
                                ))
                

        #查重
        if hdf5_sources:
            from collections import Counter 
            counter = Counter([s["file"] for s in hdf5_sources]) 
            dup = {k:v for k,v in counter.items() if v>1}
            assert len(dup)==0, f"检测到{len(dup)}组重复项：{dup}"
            
        if tfds_sources:
            from collections import Counter 
            counter = Counter([(s["directory"],s["name"],s["version"]) for s in hdf5_sources]) 
            dup = {k:v for k,v in counter.items() if v>1}
            assert len(dup)==0, f"检测到{len(dup)}组重复项：{dup}"
                
        it =  itertools.chain(
            self._generate_examples_from_hdf5(hdf5_sources),
            self._generate_examples_from_tfds(tfds_sources)
            )
        
        keys = set()
        for k,v in it:
            if k in keys: #去重
                continue
            keys.add(k)
            yield k,v
 

    
    def _generate_examples_from_hdf5(self,sources:List[ dict]) -> Iterator[Tuple[str, Any]]:
        if not sources:
            return iter(()) 

        import random  
        random.shuffle(sources)  
        num_episodes = len(sources)
        from functools import partial
        #_parse_episode=partial(data_utils.parse_episode_from_hdf5, 
        #                                            image_name_map=self.image_map,
        #                                            language_instructions={language_instruction},
        #                                            image_resize=self.image_size[:2]
        #                                            )  
        _parse_episode=data_utils.parse_episode_from_hdf5
        if self.max_workers==1:
            for episode_file in tqdm(sources, total=num_episodes, desc="提取HDF5数据"):
                try:
                    yield _parse_episode(episode_file)
                except Exception as e:
                    stack = traceback.format_exc()

                    logging.error(f"读取文件失败:{episode_file},detail: {stack}")
            
        else:
            from multiprocessing import Pool 
            #from multiprocessing.dummy import Pool
            with Pool(processes=self.max_workers) as pool:    
                results = pool.imap_unordered(_parse_episode,sources)  
                for result in tqdm(results, total=num_episodes, desc="提取HDF5数据"):  
                    yield result  
        
    def _generate_examples_from_tfds(self,sources:List[ dict]) -> Iterator[Tuple[str, Any]]:   
        def _parse_example(episode_):
            # load raw data --> this should change for your dataset
            #data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case
            #data = _load_h5_data(episode_path)
            
            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            
            
            episode_path = episode_['episode_metadata']['file_path'].numpy()
            episode = []
            for step in episode_['steps']:
                # compute Kona language embedding
                #language_embedding = self._embed([step['language_instruction']])[0].numpy()
                #print(f"step={step}")
                episode.append({
                    'observation': {
                        k:v.numpy() for k,v in step['observation'].items()
                    },
                    'action': step['action'].numpy(),
                    'discount': step['discount'].numpy(),
                    'reward': step['reward'].numpy(),
                    'is_first': step['is_first'].numpy(),
                    'is_last': step['is_last'].numpy(),
                    'is_terminal': step['is_terminal'].numpy(),
                    'language_instruction': step['language_instruction'].numpy(),
                    #'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

            
        def merge_and_shuffle_datasets(sources,shuffle_buffer_size=100):
            
            if not sources:
                return None, 0
            
            datasets=[]
            #for name in dataset_names:
            num_examples = 0
            for s in sources:
                directory, name, version = s.get("directory"),s.get("name"),s.get("version","1.0.0")
                max_example_num = s.get("max_example_num",None)
                builder = tfds.builder_from_directory(os.path.join(directory,name,version))
                
                info = builder.info  
  
                # 获取训练集的大小  
                _num_examples = info.splits['all'].num_examples 
                
                data=builder.as_dataset(
                        split='all', 
                )
                if max_example_num:
                    _num_examples = min(_num_examples,max_example_num)
                    data = data.take(max_example_num)
                datasets.append(data.prefetch(8))
                num_examples += _num_examples

            # 合并所有数据集  
            merged_dataset = datasets[0]  
            for ds in datasets[1:]:  

                merged_dataset = merged_dataset.concatenate(ds)  

            # 打乱数据集 
            #merged_dataset = merged_dataset.shuffle(shuffle_buffer_size) 

            return merged_dataset,num_examples

        merged_dataset,num_examples = merge_and_shuffle_datasets(sources)
        if not num_examples:
            return iter(())

        dup = set()
        start_fetch = time.time()
        for episode in tqdm(merged_dataset,total=num_examples,desc="提取TFDS数据"):
            cost_fetch = time.time()-start_fetch
            start_parse = time.time()
            example=_parse_example(episode)
            cost_parse = time.time()-start_parse
            #episode_path = episode['episode_metadata']['file_path'].numpy()
            if example[0] in dup:
                continue
            else:
                dup.add(example[0])
            start_yield = time.time()
            yield example
            cost_yield = time.time()-start_yield
            cost = dict(
                fetch=cost_fetch,
                parse=cost_parse,
                outer=cost_yield,
            )
           # print(f"{cost}")

            start_fetch = time.time()
        pass

def make_builder(
    dataset_name:str,
    overwrite: bool=False,
    fail_if_exists: bool=False,
    **builder_kwargs,
) -> tfds.core.DatasetBuilder:
  """Builder factory, eventually deleting pre-existing dataset."""
  builder_cls = type(dataset_name,(BaseDatasetBuilder,),{})
  builder = builder_cls(**builder_kwargs)  # pytype: disable=not-instantiable
  data_exists = builder.data_path.exists()
  if fail_if_exists and data_exists:
    raise RuntimeError(
        'The `fail_if_exists` flag was True and '
        f'the data already exists in {builder.data_path}'
    )
  if overwrite and data_exists:
    builder.data_path.rmtree()  # Delete pre-existing data
    # Re-create the builder with clean state
    builder = builder_cls(**builder_kwargs)  # pytype: disable=not-instantiable
  return builder