#!/usr/bin/python

from PIL import Image
from PIL import ImageTk
import tkinter
import glob
import sys
import os.path
import os

image_list = [os.path.join("/home/pi/fotos/previews",fn) for fn in next(os.walk("/home/pi/fotos/previews"))[2]]
sorted_imagelist = sorted(image_list, key=str.swapcase, reverse=True)

current = 0

def move(delta):
    global current, sorted_imagelist
    if not (0 <= current - delta < len(sorted_imagelist)):
        tkMessageBox.showinfo('End', 'No more image.')
        return
    current -= delta
    image = Image.open(sorted_imagelist[current])
    photo = ImageTk.PhotoImage(image)
    label['image'] = photo
    label.photo = photo


root = tkinter.Tk()

root.configure(background="#eee")

label = tkinter.Label(root, compound=tkinter.TOP, bg="#eee")
label.pack()
label.place(x=90, y=30)

frame = tkinter.Frame(root, bg="#eee")
frame.pack()

tkinter.Button(frame, text='Refresh', height=10, width=25, command=root.update).pack(side=tkinter.LEFT)
tkinter.Button(frame, text='Previous picture', height=10, width=25, command=lambda: move(-1)).pack(side=tkinter.LEFT)
tkinter.Button(frame, text='Next picture', height=10, width=25, command=lambda: move(+1)).pack(side=tkinter.LEFT)
tkinter.Button(frame, text='Quit', height=10, width=25, command=root.quit).pack(side=tkinter.LEFT)

move(0)

root.attributes('-fullscreen', True)
root.mainloop()