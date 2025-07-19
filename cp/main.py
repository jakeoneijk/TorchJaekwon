from tqdm import tqdm

from torch_jaekwon.controller import Controller

from h_params import HParams
HParams()

def run():
    controller = Controller()
    controller.run()

def run_multiple(config_path_list:list) -> None:
    for config_path in tqdm(config_path_list, desc='config_name_list'):
        HParams().set_config(config_path)
        controller = Controller()
        controller.run()

if __name__ == '__main__':
    run()
    #run_multiple([])