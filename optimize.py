# source: https://stackoverflow.com/questions/10607468/how-to-reduce-the-image-file-size-using-pil
from os import walk
from PIL import Image

path = "img/kever2/"
img = "test_comp.jpg"
path_destination = "img/kever2/result/"

model_name = "kever"
start_counter = 22


# origineel gsm gil: 3024x4032  3,17 MB
# compressie fb: 720x960  62 kB

# Read all files in dir
f = []
for (dirpath, dirnames, filenames) in walk(path):
    f.extend(filenames)
    break

print(f)
for img_file in f:

    foo = Image.open(path + img_file)
    foo = foo.resize((720, 960), Image.ANTIALIAS) # I downsize the image with an ANTIALIAS filter (gives the highest quality)

    dest = path_destination + model_name + str(start_counter) + ".jpg"
    print(dest)
    foo.save(dest, quality=95)
    # foo.save(path_destination+ "image_scaled_opt.jpg",optimize=True,quality=95)
    start_counter = start_counter+1

