import os
class Git:
    def add_all_ext(self, root:str='.', ext_list:list = ['.py','.yaml'])->None:
        for ext in ext_list:
            for root_dir, dir_list, file_list in os.walk(root):
                for file_name in file_list:
                    if os.path.splitext(file_name)[-1] == ext:
                        os.system(f'git add {root_dir}/{file_name}')

if __name__ == '__main__':
    Git().add_all_ext()