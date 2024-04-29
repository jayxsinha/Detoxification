#!/bin/bash

LOCAL_DIR="./"  # Your local directory
REMOTE_USER="jsinha_umass_edu"  # Your remote username
REMOTE_HOST="unity.rc.umass.edu"  # Your remote server
REMOTE_DIR="/home/jsinha_umass_edu/Guided-Detoxification"  # Remote directory path

# Rsync command
# rsync -az  -e "ssh -i /Users/jaysinha/.ssh/unity" "$LOCAL_DIR" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"
rsync -az --exclude='output' -e "ssh -i /Users/jaysinha/.ssh/unity" "$LOCAL_DIR" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR"