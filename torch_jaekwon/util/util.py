from typing import Literal, List
import os, sys
import psutil
import re
import torch
import importlib.util

PURPLE = '\033[95m'
CYAN = '\033[96m'
DARKCYAN = '\033[36m'
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'
END = '\033[0m'

def get_resource_usage(
    verbose=True,
    min_available_ram_mb:float = 1000.0,
)-> dict:
    log_dict = dict()
    process = psutil.Process(os.getpid())
    log_dict['ram_usage_mb'] = process.memory_info().rss / 1024 ** 2
    virtual_mem = psutil.virtual_memory()
    log_dict['ram_available_mb'] = virtual_mem.available / 1024 ** 2
    log_dict['ram_total_mb'] = virtual_mem.total / 1024 ** 2

    log_dict['cpu_usage_percent'] = process.cpu_percent(interval=0.1)

    if torch.cuda.is_available():
        log_dict['cuda_allocated_mb'] = torch.cuda.memory_allocated() / 1024 ** 2
        log_dict['cuda_reserved_mb'] = torch.cuda.memory_reserved() / 1024 ** 2

    if verbose: 
        message = [f'[{key}]: {value:.2f}'for key, value in log_dict.items()]
        message = ' | '.join(message)
        log(message, msg_type='info')

    if min_available_ram_mb is not None and log_dict['ram_available_mb'] < min_available_ram_mb:
        log(f"Available RAM ({log_dict['available_ram_mb']:.2f} MB) below threshold ({min_available_ram_mb} MB). Exiting.", msg_type='error')
        sys.exit(1)
    return log_dict

def log(text:str, msg_type:Literal['info', 'success', 'warning', 'error'] = None, prefix:str = '') -> None:
    template_dict:dict = {
        'info': {
            'color': BOLD + BLUE,
            'prefix': '[Info]: '
        },
        'success': {
            'color': BOLD + GREEN,
            'prefix': '[Success]: '
        },
        'warning': {
            'color': BOLD + YELLOW,
            'prefix': '[Warning]: '
        },
        'error': {
            'color': BOLD + RED,
            'prefix': '[Error]: '
        }
    }
    color:str = template_dict.get(msg_type, {}).get('color', '')
    prefix:str = prefix + template_dict.get(msg_type, {}).get('prefix', '')
    print(f"{color + prefix + text + END}")

def get_num_in_str(text:str) -> List[int]:
    return [int(n) for n in re.findall(r'\d+', text)]

def set_sys_path_to_parent_dir(
    file:str, # __file__
    depth_to_dir_from_file:int = 1,
) -> None:
    dir:str = get_ancestor_dir_path(file, depth_to_dir_from_file)
    sys.path[0] = os.path.abspath(dir)

def get_ancestor_dir_path(
    file:str, # __file__
    depth_to_dir_from_file:int = 1,
) -> str:
    dir : str = os.path.abspath(os.path.dirname(file))
    for _ in range(depth_to_dir_from_file): dir = os.path.dirname(dir)
    return dir

def system(command:str) -> None:
    result_id:int = os.system(command)
    assert result_id == 0, f'[Error]: Something wrong with the command [{command}]'

def cp(src_path:str, dst_path:str, inside_dir:bool = False) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.isdir(src_path):
        src_path = os.path.join(src_path, '*') if inside_dir else f"'{src_path}'"
        system(f"cp -r {src_path} '{dst_path}'") 
    else:
        system(f"cp '{src_path}' '{dst_path}'")

def wget(link:str, save_path:str = None, save_dir:str = None) -> None:
    command:str = f'wget -c -t 0 --retry-connrefused --waitretry=5 {link}'
    if save_path is not None:
        make_parent_dir(save_path)
        command += f' -O {save_path}'
    elif save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        command += f' -P {save_dir}'
    system(command)

def unzip(file_path:str, unzip_dir:str) -> None:
    os.makedirs(unzip_dir, exist_ok=True)
    file_name:str = file_path.split('/')[-1]
    if '.tar.gz' in file_name:
        system(f'''tar -xzvf {file_path} -C {unzip_dir}''')
    else:
        system(f'''unzip {file_path} -d {unzip_dir}''')

def norm_path(file_path:str) -> str:
    if file_path[0] not in ['/', '.']:
        return f"./{file_path}"
    return file_path

def make_parent_dir(file_path:str) -> None:
    os.makedirs(os.path.dirname(norm_path(file_path)), exist_ok=True)

def is_package_installed(package_name: str) -> bool:
    if importlib.util.find_spec(package_name) is not None:
        return True
    else:
        return False