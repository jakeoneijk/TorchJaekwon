SERVER="id@domain"
PORT=22
TARGET_DIR=""
SRC_DIR=""

rsync -avz --partial --append-verify --progress -e "ssh -p $PORT" "$SRC_DIR" "$SERVER:$TARGET_DIR"
#rsync -avz --partial --append-verify --progress -e "ssh -p $PORT" "$SERVER:$SRC_DIR" "$TARGET_DIR"

#CONFIG_FILE="$HOME/.ssh/config"
#rsync -avz --partial --append-verify --progress -e "ssh -F $CONFIG_FILE" "$SRC_DIR" "$SERVER:$TARGET_DIR"
#rsync -avz --partial --append-verify --progress -e "ssh -F $CONFIG_FILE" "$SERVER:$SRC_DIR" "$TARGET_DIR"