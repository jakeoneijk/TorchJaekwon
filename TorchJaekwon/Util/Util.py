import os, sys

class Util:
    @staticmethod
    def set_sys_path_to_parent_dir(file:str, # __file__
                                   depth_to_dir_from_file:int = 1,
                                   ) -> None:
        dir : str = os.path.abspath(os.path.dirname(file))
        for _ in range(depth_to_dir_from_file): dir = os.path.dirname(dir)
        sys.path[0] = os.path.abspath(dir)
    
    @staticmethod
    def system(command:str) -> None:
        result_id:int = os.system(command)
        assert result_id == 0, f'[Error]: Something wrong with the command [{command}]'
    
    @staticmethod
    def wget(link:str, save_dir:str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        Util.system(f'wget {link} -P {save_dir}')
    
    @staticmethod
    def unzip(file_path:str, unzip_dir:str) -> None:
        os.makedirs(unzip_dir, exist_ok=True)
        file_name:str = file_path.split('/')[-1]
        if '.tar.gz' in file_name:
            Util.system(f'''tar -xzvf {file_path} -C {unzip_dir}''')
        else:
            Util.system(f'''unzip {file_path} -d {unzip_dir}''')