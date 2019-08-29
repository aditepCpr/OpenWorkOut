import tkinter as tk
from tkinter import *
from tkinter import Button, Menu
from tkinter import filedialog
from tkinter import ttk
import OpenWorkout as Owk

selectionnFilename = None


class MainPage():
    def __init__(self, root):
        self.root = root
        self.pageMain()
        self._createMenu()
        self.selectionnFilename = selectionnFilename

    def pageMain(self):
        root.title("Training")
        root.geometry('800x800')
        p1 = PanedWindow(bg='black', orient=VERTICAL)
        p1.pack(fill=BOTH, expand=1)
        top = Label(p1, text="python")
        cen = Label(p1, text="top pane")
        botton = Label(p1, text="top pane")
        p1.add(top)
        p1.add(cen)
        p1.add(botton)

    def _createMenu(self):
        menubar = Menu(root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="New", command=self.donothing)
        filemenu.add_command(label="Training", command=self.pageTrain)
        filemenu.add_separator()  # --------------- #

        filemenu.add_command(label="Exit", command=self.click_exit)
        menubar.add_cascade(label="File", menu=filemenu)

        root.config(menu=menubar)
        root.mainloop()

    def click_exit(self):
        """Click a exit menu
                """
        self.root.quit()
        self.root.destroy()
        exit()

    def selection(self):
        root.filename = filedialog.askopenfilename(initialdir="/home/aditep/soflware", title="Select file",
                                                   filetypes=(("files mp4", "*.mp4"), ("all files", "*.*")))

        print(root.filename)

    def donothing(self):
        filewin = Toplevel(root)
        button = Button(filewin, text="Do nothing button")
        button.pack()

    def pageTrain(self):
        def Train():
            label1.configure(text=comboExs.get())
            owk = Owk.OpenWorkpout(root.filename)
            owk._OpenCVpose()

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
        bBrowse = Button(fileTrain, text=' Browse ', bd=3, font=('', 10), padx=5, pady=5, command=self.selection)
        label1 = Label(fileTrain, text="")

        print(dict(comboExs))
        pwTrain1.add(comboExs)
        pwTrain1.add(bBrowse)
        pwTrain1.add(label1)
        pwTrain1.add(bsubmit)
        print(comboExs.current(), comboExs.get())
        print(str(bBrowse))


if __name__ == '__main__':
    root = tk.Tk()
    MainPage(root)
    root.mainloop()  # Start GUI

