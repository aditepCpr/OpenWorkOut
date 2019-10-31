import tkinter as tk
from tkinter import *
from tkinter import Button, Menu
from tkinter import filedialog
from tkinter import ttk
import OpenWorkout as Owk
from Predict_Data import predict_knn,predict_DecisionTree,predict_Lori,predict_RandomForest,predict_Svc
from TrainingModel import training_DecisionTree,training_knn,training_Lori,training_RandomForest,training_Svc
from RemoveJson import removeJson
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
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
        root.geometry('800x400')
        self.framePredictVdo(root)
        self.framePredictLive(root)
        self.framePredictData(root)
        # predictData(root)

    def framePredictVdo(self,root):
        def preDictVdo():
            label1.configure(text=comboExs.get())
            print(comboExs.get())
            owk = Owk.OpenWorkpout(root.filename, 'cam')
            owk._OpenCVpose()

        PrframeVdo = Frame(root, bd="3", relief=GROOVE, padx=10, pady=10, bg='snow')
        PrframeVdo.pack()
        labelTop = tk.Label(PrframeVdo,text="Choose your exercise")
        value = [
            "Push Ups",
            "Squat",
            "Deadlift",
            "Dumbbell Shoulder Press",
            "Barbell Curl"]

        comboExs = ttk.Combobox(PrframeVdo, values=value)
        comboExs.current(1)
        bBrowse = Button(PrframeVdo, text=' Browse ', bd=3, font=('', 10), padx=5, pady=5, command=self.selection)
        bPredictVdo = Button(PrframeVdo, text="Predict", command=preDictVdo)
        bremoveJson = Button(PrframeVdo, text="clear data", command=removeJson)
        # btrain = Button(PrframeVdo, text="Training", command=train)


        label1 = Label(PrframeVdo, text="")
        labelTop.pack(side=TOP)
        comboExs.pack(side=LEFT)
        bBrowse.pack(side=LEFT)
        bPredictVdo.pack(side=TOP)
        bremoveJson.pack(side=BOTTOM)
        # btrain.pack()


    def framePredictLive(self,root):
        Prframelive = Frame(root, bd="3", relief=GROOVE, padx=10, pady=10)
        top = Label(Prframelive, text="python")
        cen = Label(Prframelive, text="OpenWorkOut by Aditep  campira")
        botton = Label(Prframelive, text="top pane")
        blive = Button(Prframelive, text=' live ', bd=3, font=('', 10), padx=5, pady=5, command=self.pageLive)
        top.pack()
        cen.pack()
        botton.pack()
        blive.pack()
        Prframelive.pack()
        # blive.configure(text='Live Start...')


    def framePredictData(self,root):
        Prframe = Frame(root, bd="3", relief=GROOVE, padx=10, pady=10)
        Prframe.pack(side=TOP)
        label1 = Label(Prframe, text="Predict Data").grid(row=0,column=0)
        bshow = Button(Prframe, text="k-nearest neighbors", command=predict_knn).grid(row=1,column=0)
        bshow2 = Button(Prframe, text="DecisionTree", command=predict_DecisionTree).grid(row=1,column=1)
        bshow3 = Button(Prframe, text="RandomForest", command=predict_RandomForest).grid(row=1,column=2)
        bshow4 = Button(Prframe, text="SVC", command=predict_Svc).grid(row=1,column=3)
        bshow5 = Button(Prframe, text="LogisticRegression", command=predict_Lori).grid(row=1,column=4)



    def _createMenu(self):
        menubar = Menu(root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="New", command=self.donothing)
        filemenu.add_command(label="Input Data", command=self.input_Data)
        filemenu.add_command(label="Predict_ShowData", command=self.pageShow)
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
        fileshow.title("Predict_ShowData")
        pwshow = PanedWindow(fileshow, bg='red', orient=VERTICAL)
        pwshow.pack(fill=BOTH, expand=1)
        bshow = Button(fileshow, text="Knn",command=predict_knn)
        bshow2 = Button(fileshow, text="DecisionTree",command=predict_DecisionTree)
        bshow3 = Button(fileshow,  text="RandomForest",command=predict_RandomForest)
        bshow4 = Button(fileshow,  text="SVC",command=predict_Svc)
        bshow5 = Button(fileshow,  text="LogisticRegression",command=predict_Lori)
        pwshow.add(bshow)
        pwshow.add(bshow2)
        pwshow.add(bshow3)
        pwshow.add(bshow4)
        pwshow.add(bshow5)


    def input_Data(self):
        print('input_Data')
        def inputData():
            label1.configure(text=comboExs.get())
            print(comboExs.get())
            owk = Owk.OpenWorkpout(root.filename, comboExs.get())
            owk._OpenCVpose()

        def train():
            training_knn()
            training_Svc()
            training_RandomForest()
            training_Lori()
            training_DecisionTree()

        fileTrain = Toplevel(root)
        fileTrain.geometry('300x200')
        fileTrain.title("Input Data")
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
        binputData = Button(fileTrain, text="Click Here", command=inputData)
        btrain = Button(fileTrain, text="Training", command=train)
        bBrowse = Button(fileTrain, text=' Browse ', bd=3, font=('', 10), padx=5, pady=5, command=self.selection)

        label1 = Label(fileTrain, text="")

        # print(dict(comboExs))
        pwTrain1.add(comboExs)
        pwTrain1.add(bBrowse)
        pwTrain1.add(label1)
        pwTrain1.add(binputData)
        pwTrain1.add(btrain)

        # print(comboExs.current(), comboExs.get())
        # print(str(bBrowse))
    def train_Data(self):
        print('train_Data')
        def inputData():
            label1.configure(text=comboExs.get())
            print(comboExs.get())
            owk = Owk.OpenWorkpout(root.filename, comboExs.get())
            owk._OpenCVpose()

        def train():
            training_Svc()
            training_RandomForest()
            training_Lori()
            training_knn()
            training_DecisionTree()

        fileTrain = Toplevel(root)
        fileTrain.geometry('300x200')
        fileTrain.title("Input Data")
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
        binputData = Button(fileTrain, text="Click Here", command=inputData)
        btrain = Button(fileTrain, text="Training", command=train)
        bBrowse = Button(fileTrain, text=' Browse ', bd=3, font=('', 10), padx=5, pady=5, command=self.selection)

        label1 = Label(fileTrain, text="")

        # print(dict(comboExs))
        pwTrain1.add(comboExs)
        pwTrain1.add(bBrowse)
        pwTrain1.add(label1)
        pwTrain1.add(binputData)
        pwTrain1.add(btrain)

if __name__ == '__main__':
    root = tk.Tk()
    # content = Frame(root)
    MainPage(root)
    root.mainloop()  # Start GUI

