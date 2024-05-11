git init
touch .gitignore
echo __pycache__ >> .gitignore
echo wandb >> .gitignore
echo *.wav >> .gitignore
echo *.png >> .gitignore
echo *.pkl >> .gitignore
echo *.pt >> .gitignore
echo *.pth >> .gitignore
echo *.ckpt >> .gitignore

source ./Script/2_git_add_all_ext.sh py
source ./Script/2_git_add_all_ext.sh sh
source ./Script/2_git_add_all_ext.sh yaml
source ./Script/2_git_add_all_ext.sh json
source ./Script/2_git_add_all_ext.sh txt