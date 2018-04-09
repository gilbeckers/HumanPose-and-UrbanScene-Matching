# source: https://stackoverflow.com/questions/10607468/how-to-reduce-the-image-file-size-using-pil
from os import walk
import os

path = "img/bull/"
new_name = 'bull'

start_counter = 1

# Read all files in dir
f = []
for (dirpath, dirnames, filenames) in walk(path):
    f.extend(filenames)
    break

for img_file in f:
    print(path + img_file)
    os.rename(path + img_file, path + new_name + str(start_counter) + '.jpg')
    start_counter = start_counter+1


