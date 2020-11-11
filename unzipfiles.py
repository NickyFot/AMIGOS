import os
import subprocess
import logging
import json

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    for root, dirs, filenames in os.walk('/home/niki/Datasets/AMIGOS/Videos/'):
        for filename in filenames:
            filename = os.path.join(root, filename)
            if not filename.endswith('.zip'):
                continue
            os.system("7za x " + filename + ' -aos')
