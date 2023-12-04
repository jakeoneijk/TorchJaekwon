import os, sys

class Util:
    @staticmethod
    def set_sys_path_to_parent_dir(file:str, # __file__
                                   depth_to_dir_from_file:int = 1,
                                   ) -> None:
        dir : str = os.path.abspath(os.path.dirname(file))
        for _ in range(depth_to_dir_from_file): dir = os.path.dirname(dir)
        sys.path[0] = os.path.abspath(dir)