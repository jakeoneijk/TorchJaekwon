git init
touch .gitignore
echo __pycache__ >> .gitignore
echo wandb >> .gitignore
echo *.wav >> .gitignore
echo *.pth >> .gitignore
echo *.png >> .gitignore
echo *.pkl >> .gitignore

git add *.py
git add */*.py
git add */*/*.py
git add */*/*/*.py
git add */*/*/*/*.py
git add */*/*/*/*/*.py
git add */*/*/*/*/*/*.py
git add */*/*/*/*/*/*/*.py
git add */*/*/*/*/*/*/*/*.py

git add *.yaml
git add */*.yaml
git add */*/*.yaml
git add */*/*/*.yaml
git add */*/*/*/*.yaml
git add */*/*/*/*/*.yaml
git add */*/*/*/*/*/*.yaml
git add */*/*/*/*/*/*/*.yaml
git add */*/*/*/*/*/*/*/*.yaml