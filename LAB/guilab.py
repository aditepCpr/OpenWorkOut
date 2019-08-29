
import tkinter as tk
from tkinter import ttk
from tkinter import *
from tkinter import Button
from tkinter import filedialog
import OpenWorkout2 as owk1
def selection():
    root.filename = filedialog.askopenfilename(initialdir="/home/aditep/soflware", title="Select file",
                                               filetypes=(("files mp4", "*.mp4"), ("all files", "*.*")))

    print(root.filename)
    # owk = Owk2.OpenWorkpout2(root.filename)
    # self.root.quit()
    # owk._OpenCVpose()


def pageTrain(root):
    def Train():
        label1.configure(text=comboExs.get())

    fileTrain = Toplevel(root)
    fileTrain.geometry('300x200')
    fileTrain.title("Training")
    pwTrain1 = PanedWindow(fileTrain, bg='red', orient=VERTICAL)
    pwTrain1.pack(fill=BOTH, expand=1)
    labelTop = tk.Label(pwTrain1,
                        text="Choose your exercise")
    pwTrain1.add(labelTop)
    value = [
        "Push Ups",
        "Squat",
        "Deadlift",
        "Dumbbell Shoulder Press",
        "Barbell Curl"]

    comboExs = ttk.Combobox(pwTrain1, values=value)
    comboExs.current(1)
    bsubmit = Button(fileTrain, text="Click Here", command=Train)
    bBrowse = Button(fileTrain, text=' Browse ', bd=3, font=('', 10), padx=5, pady=5, command=selection)
    label1 = Label(fileTrain, text="")

    print(dict(comboExs))
    pwTrain1.add(comboExs)
    pwTrain1.add(bBrowse)
    pwTrain1.add(label1)
    pwTrain1.add(bsubmit)
    print(comboExs.current(), comboExs.get())
    print(str(bBrowse))

if __name__ == '__main__':
    # root = tk.Tk()
    # pageTrain(root)
    # root.mainloop()
    owk = owk1.OpenWorkpout2('/home/aditep/soflware/OpenWorkOut/vdo/BarbellCurl/BarbellCurl1.mp4')
    owk._OpenCVpose()