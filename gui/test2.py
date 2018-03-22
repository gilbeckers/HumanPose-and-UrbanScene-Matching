import os
from os import walk
from PIL import Image
try:
  import Tkinter
except ImportError:
  import tkinter as Tkinter

root = Tkinter.Tk()
L = Tkinter.Listbox(selectmode=Tkinter.SINGLE)
gifsdict = {}

dirpath = '../img'
f = []
for (dirpath, dirnames, filenames) in walk(dirpath):
    f.extend(filenames)
    break


for gifname in f:
    print(gifname)
    # if not gi
    #    continue
    gifpath = os.path.join(dirpath, gifname)
    gifpath = "../img/" + gifname
    print(gifpath)
    gif = Image.open(gifpath) #Tkinter.PhotoImage(file=gifpath)
    gifsdict[gifname] = gif
    L.insert(Tkinter.END, gifname)

L.pack()
img = Tkinter.Label()
img.pack()

def list_entry_clicked(*ignore):
    imgname = L.get(L.curselection()[0])
    img.config(image=gifsdict[imgname])

L.bind('<ButtonRelease-1>', list_entry_clicked)

root.mainloop()