import tkinter as tk
from tkinter import *
from tkinter import Button, Menu
from tkinter import filedialog
from tkinter import ttk
import OpenWorkout as Owk
from Predict_Data import predict_knn,predict_DecisionTree,predict_Mlpc,predict_RandomForest,predict_Svc
from TrainingModel import training_DecisionTree,training_knn,training_mlpc,training_RandomForest,training_Svc,training_percep
from RemoveJson import removeJson
from TrainingModel_Exercise import training_knnEx,training_DecisionTreeEx,training_mlpcEx,training_percepEx,training_RandomForestEx,training_SvcEx,data
from StackData import DataModel
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
        self.framePredictLive(root)
        self.framePredictVdo(root)
        self.framePredictData(root)
        # predictData(root)

    def framePredictVdo(self,root):
        def preDictVdo():
            label1.configure(text=comboExs.get())
            print(comboExs.get())
            owk = Owk.OpenWorkpout(root.filename, 'cam')
            owk._OpenCVpose()
        def trainingEx():
            training_knnEx(comboExs.get())
            training_DecisionTreeEx(comboExs.get())
            training_SvcEx(comboExs.get())
            training_RandomForestEx(comboExs.get())
            training_mlpcEx(comboExs.get())

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
        bTrainingEx = Button(PrframeVdo, text="trainingEx", command=trainingEx)
        # btrain = Button(PrframeVdo, text="Training", command=train)


        label1 = Label(PrframeVdo, text="")
        labelTop.pack(side=TOP)
        comboExs.pack(side=LEFT)
        bBrowse.pack(side=LEFT)
        bPredictVdo.pack(side=TOP)
        bremoveJson.pack(side=BOTTOM)
        bTrainingEx.pack(side=BOTTOM)
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
        bshow5 = Button(Prframe, text="MLPClassifier", command=predict_Mlpc).grid(row=1,column=4)



    def _createMenu(self):
        menubar = Menu(root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="New", command=self.donothing)
        filemenu.add_command(label="Train Model", command=self.input_Data)
        filemenu.add_command(label="Predict_TypeExercise", command=self.framePredictVdo)
        filemenu.add_command(label="Predict_Workout", command=self.framePredictVdo)
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
        fileshow.title("Show Model")
        pwshow = PanedWindow(fileshow, bg='red', orient=VERTICAL)
        pwshow.pack(fill=BOTH, expand=1)
        bshow = Button(fileshow, text="Knn",command=predict_knn)
        bshow2 = Button(fileshow, text="DecisionTree",command=predict_DecisionTree)
        bshow3 = Button(fileshow,  text="RandomForest",command=predict_RandomForest)
        bshow4 = Button(fileshow,  text="SVC",command=predict_Svc)
        bshow5 = Button(fileshow,  text="MLPClassifier",command=predict_Mlpc)
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

        def trainType():
            training_knn()
            training_Svc()
            training_RandomForest()
            training_percep()
            training_mlpc()
            training_DecisionTree()

        def trainingEx():
            training_knnEx(comboExs.get())
            training_DecisionTreeEx(comboExs.get())
            training_SvcEx(comboExs.get())
            training_RandomForestEx(comboExs.get())
            training_mlpcEx(comboExs.get())

        fileTrain = Toplevel(root)
        fileTrain.geometry('800x400')
        fileTrain.title("Training Model/Input Data")
        # pwTrain1 = PanedWindow(fileTrain, bg='red', orient=VERTICAL)
        # pwTrain1.pack(fill=BOTH, expand=1)
        Prframe = Frame(fileTrain, bd="3", relief=GROOVE, padx=100, pady=100)
        Prframe.pack(side=TOP)
        Prframe2 = Frame(fileTrain, bd="3", relief=GROOVE, padx=20, pady=20)
        Prframe2.pack(side=BOTTOM)
        labelTop = tk.Label(Prframe,
                            text="Choose your exercise for inputData and Train_workout")
        labelTop.pack()
        value = [
            "Push Ups",
            "Squat",
            "Deadlift",
            "Dumbbell Shoulder Press",
            "Barbell Curl"]

        comboExs = ttk.Combobox(Prframe, values=value)
        comboExs.current(1)
        binputData = Button(Prframe, text="Input Data", command=inputData)
        btrainType = Button(Prframe2, text="Train_Type", command=trainType)
        btrainWorkout = Button(Prframe2, text="Train_workout", command=trainingEx)
        bBrowse = Button(Prframe, text=' Browse ', bd=3, font=('', 10), padx=5, pady=5, command=self.selection)

        label1 = Label(Prframe2, text="Training Model")

        # print(dict(comboExs))
        comboExs.pack()
        bBrowse.pack()
        label1.pack(side = TOP)
        binputData.pack()
        btrainType.pack(side=LEFT)
        btrainWorkout.pack(side=RIGHT)


        # print(comboExs.current(), comboExs.get())
        # print(str(bBrowse))
    # def train_Data(self):
    #     print('train_Data')
    #     def inputData():
    #         label1.configure(text=comboExs.get())
    #         print(comboExs.get())
    #         owk = Owk.OpenWorkpout(root.filename, comboExs.get())
    #         owk._OpenCVpose()
    #
    #     def train():
    #         training_knn()
    #         training_Svc()
    #         training_RandomForest()
    #         training_percep()
    #         training_mlpc()
    #         training_DecisionTree()
    #
    #     fileTrain = Toplevel(root)
    #     fileTrain.geometry('800x400')
    #     fileTrain.title("Training Model/Input Data")
    #     pwTrain1 = PanedWindow(fileTrain, bg='red', orient=VERTICAL)
    #     pwTrain1.pack(fill=BOTH, expand=1)
    #     Prframe = Frame(pwTrain1, bd="3", relief=GROOVE, padx=10, pady=10)
    #     Prframe.pack(side=TOP)
    #     labelTop = tk.Label(Prframe,
    #                         text="Choose your exercise")
    #     Prframe.add(labelTop)
    #     value = [
    #         "Push Ups",
    #         "Squat",
    #         "Deadlift",
    #         "Dumbbell Shoulder Press",
    #         "Barbell Curl"]
    #
    #     comboExs = ttk.Combobox(Prframe, values=value)
    #     comboExs.current(1)
    #     binputData = Button(Prframe, text="Click Here", command=inputData)
    #     btrain = Button(Prframe, text="Training", command=train)
    #     bBrowse = Button(Prframe, text=' Browse ', bd=3, font=('', 10), padx=5, pady=5, command=self.selection)
    #
    #     label1 = Label(Prframe, text="")
    #
    #     # print(dict(comboExs))
    #     comboExs.pack()
    #     bBrowse.pack()
    #     label1.pack()
    #     binputData.pack()
    #     btrain.pack()

if __name__ == '__main__':
    root = tk.Tk()
    # content = Frame(root)
    MainPage(root)
    root.mainloop()  # Start GUI

