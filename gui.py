import tkinter as tk
from tkinter import *
from tkinter import Button, Menu
from tkinter import filedialog
from tkinter import ttk
import OpenWorkout as Owk


selectionnFilename = None


class MainPage():
    print('MainPage Ok...')
    def __init__(self, root):
        self.root = root
        self.pageMain()
        self._createMenu()
        self.selectionnFilename = selectionnFilename

    def pageLive(self):
        try:
            owk = Owk.OpenWorkpout(0,'cam')
            owk._OpenCVpose()
        except Exception as e:
            print(e)

    def pageMain(self):
        print('GUI Start...')
        root.title("Training")
        root.geometry('300x300')
        p1 = PanedWindow(bg='black', orient=VERTICAL)
        p1.pack(fill=BOTH, expand=1)
        top = Label(p1, text="python")
        cen = Label(p1, text="OpenWorkOut by Aditep  campira")
        botton = Label(p1, text="top pane")
        blive = Button(p1, text=' live ', bd=3, font=('', 10), padx=5, pady=5, command=self.pageLive)
        p1.add(top)
        p1.add(cen)
        p1.add(blive)

    def _createMenu(self):
        menubar = Menu(root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="New", command=self.donothing)
        filemenu.add_command(label="Training", command=self.pageTrain)
        filemenu.add_command(label="Show", command=self.pageShow)
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

    def pageShow(self):
        fileshow = Toplevel(root)
        fileshow.geometry('300x200')
        fileshow.title("ShowData")
        pwshow = PanedWindow(fileshow, bg='red', orient=VERTICAL)
        pwshow.pack(fill=BOTH, expand=1)
        bshow = Button(fileshow, text="1",command=sd.show)
        bshow2 = Button(fileshow, text="2",command=sd.show2)
        bshow3 = Button(fileshow,  text="3",command=sd.show3)
        bshow4 = Button(fileshow,  text="4",command=sd.show4)
        pwshow.add(bshow)
        pwshow.add(bshow2)
        pwshow.add(bshow3)
        pwshow.add(bshow4)


    def pageTrain(self):
        print('Train')
        def Train():
            label1.configure(text=comboExs.get())
            print(comboExs.get())
            owk = Owk.OpenWorkpout(root.filename,comboExs.get())
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

        # print(dict(comboExs))
        pwTrain1.add(comboExs)
        pwTrain1.add(bBrowse)
        pwTrain1.add(label1)
        pwTrain1.add(bsubmit)

        # print(comboExs.current(), comboExs.get())
        # print(str(bBrowse))


if __name__ == '__main__':
    root = tk.Tk()
    MainPage(root)
    root.mainloop()  # Start GUI

