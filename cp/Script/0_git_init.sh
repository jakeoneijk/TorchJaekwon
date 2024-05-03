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

source 1_git_add_all_ext.sh py
source 1_git_add_all_ext.sh sh
source 1_git_add_all_ext.sh yaml