# Torch Jaekwon
This is my personal repository for the efficiency of my research.

## Set up
* Clone the repository.
    ```
    git clone https://github.com/jakeoneijk/TorchJaekwon
    ```

* Install TorchJaekwon
    ```
    source ./cp/Script/1_install_torchjk.sh
    ```

* Copy the files and directories in 'cp' to the new project directory.
    ```
    .
    └── TorchJaekwon
        └── cp
    ```

## Usage
### Basic
```shell
> python Main.py [-s STAGE_NAME]

arguments:
    -s STAGE_NAME, --stage STAGE_NAME
    #STAGE of the system. choices = ['preprocess', 'train', 'inference', 'evaluate']
```

