#!/bin/bash
  
check_del_file(){
        if [ ! -f "$1" ]; then
                echo -e "\n\nError: $1 does not exist."
                exit 1
        fi
        rm $1
}
