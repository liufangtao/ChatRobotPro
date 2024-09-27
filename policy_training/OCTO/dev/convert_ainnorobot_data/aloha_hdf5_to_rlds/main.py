import json
import os
import sys
import argparse
import hydra
from omegaconf import OmegaConf
import yaml
 
@hydra.main(config_path='cfgs', config_name='config') 
def main(cfg):

    import logging  
  
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s]: %(message)s')  
  

    print(f"config=\n{yaml.dump(OmegaConf.to_container(cfg, resolve=True), allow_unicode=True)}") 

    from base_dataset_builder import make_builder 
    dataset_builder = make_builder(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.dataset.output_dir, 
        source=cfg.dataset.source,
        observatoin_info=cfg.dataset.observations,
        actions_info=cfg.dataset.actions,
        overwrite=cfg.get("overwrite",False),
        max_workers=cfg.get("workers",1),
    )
    # 构建和打包数据集  
    dataset_builder.download_and_prepare() 

    config_file = os.path.join(cfg.dataset.output_dir,cfg.dataset.name+'.yaml')
    with open(config_file, "w") as f:  
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f, allow_unicode=True)

  
if __name__ == '__main__': 
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '' #不要占用GPU  
    main()