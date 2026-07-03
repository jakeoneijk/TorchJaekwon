import os
import argparse
from torch_jaekwon.util import util
from torch_jaekwon.path import TORCH_JAEKWON_PATH, ARTIFACTS_DIRS

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dest_dir_path",
        type=str,
        required=False,
        default='./test_project',
        help="",
    )
    args = parser.parse_args()

    dest_dir_path:str = args.dest_dir_path
    os.makedirs(dest_dir_path, exist_ok=True)
    util.cp(src_path=f'{os.path.dirname(TORCH_JAEKWON_PATH)}/cp/.', dst_path=f'{dest_dir_path}')

    depth_threshold:int = 4
    excluded_name_list:list = ['__pycache__', 'external', 'model/', 'loss/', 'util/']
    for root, dirs, files in os.walk(TORCH_JAEKWON_PATH):
        if os.path.relpath(root, TORCH_JAEKWON_PATH).count(os.sep) >= depth_threshold: continue
        if any([excluded_name in root for excluded_name in excluded_name_list]): continue
        os.makedirs(os.path.join(dest_dir_path, os.path.relpath(root, TORCH_JAEKWON_PATH)), exist_ok=True)

    # per-stage class dirs to scaffold in the new project (was torch_jaekwon.path.CLASS_DIRS, now convention-only)
    class_dir_list:list = [
        'data/dataset_manager', 'data/preprocess', 'model', 'train/trainer', 'data/dataset',
        'train/optimizer/scheduler', 'train/loss', 'inference', 'evaluate/evaluator',
    ]
    for dir_path in class_dir_list + list(ARTIFACTS_DIRS.__dict__.values()):
        os.makedirs(os.path.join(dest_dir_path, dir_path.replace('./','')), exist_ok=True)
        
    util.log(f"Project initialized at {dest_dir_path}", msg_type='success')