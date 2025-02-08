from tqdm import tqdm

from TorchJaekwon.Controller import Controller

from HParams import HParams
HParams()

def run():
    controller = Controller()
    controller.run()

def run_multiple(config_name_list:list) -> None:
    for config_name in tqdm(config_name_list, desc='config_name_list'):
        HParams().set_config(config_name)
        controller = Controller()
        controller.run()

if __name__ == '__main__':
    run()
    #run_multiple([])