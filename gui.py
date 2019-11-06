import tkinter as tk
from tkinter import *
from tkinter import Button, Menu
from tkinter import filedialog
from tkinter import ttk
import OpenWorkout as Owk
from Predict_Data import predict_knn, predict_DecisionTree, predict_Mlpc, predict_RandomForest, predict_Svc
from TrainingModel import training_DecisionTree, training_knn, training_mlpc, training_RandomForest, training_Svc, \
    training_percep
from RemoveJson import removeJson
from TrainingModel_Exercise import training_knnEx, training_DecisionTreeEx, training_mlpcEx, training_percepEx, \
    training_RandomForestEx, training_SvcEx, data
from tkinter import messagebox

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
            owk = Owk.OpenWorkpout(0, 'cam', None)
            owk._OpenCVpose()
        except Exception as e:
            messagebox.showinfo("Error", e)
            print(e)

    def pageMain(self):
        print('GUI Start...')
        root.title("Training")
        root.geometry('800x400')
        self.framePredictLive(root)
        self.framePredictTypeWorkOutData(root)
        self.framePredictVdo(root)

        # predictData(root)

    def framePredictVdo(self, root):
        def preInputVdoData():
            # messagebox.showinfo("PredictVdo",'Predict : " '+ comboExs.get()+'" :  Coming soon')
            label1.configure(text=comboExs.get())
            print(comboExs.get())
            owk = Owk.OpenWorkpout(0, 'predictVdo', comboExs.get())
            owk._OpenCVpose()

        # filePredictVdo = Toplevel(root)
        # filePredictVdo.geometry('800x400')
        # filePredictVdo.title("Predict Vdo")
        PrframeVdo = Frame(root, bd="3", relief=GROOVE, padx=10, pady=10, bg='snow')
        PrframeVdo.pack()
        labelTop = tk.Label(PrframeVdo, text="Choose your exercise", bg='snow')
        value = [
            "Push Ups",
            "Squat",
            "Deadlift",
            "Dumbbell Shoulder Press",
            "Barbell Curl"]

        comboExs = ttk.Combobox(PrframeVdo, values=value)
        comboExs.current(1)
        # bBrowse = Button(PrframeVdo, text=' Browse ', bd=3, font=('', 10), padx=5, pady=5, command=self.selection)
        bPredictVdo = Button(PrframeVdo, text="Predict", command=preInputVdoData)
        # bInputVdoData = Button(PrframeVdo, text="import Data", command=preInputVdoData)
        # bremoveJson = Button(PrframeVdo, text="clear data", command=removeJson)
        # btrain = Button(PrframeVdo, text="Training", command=train)

        label1 = Label(PrframeVdo, text="")
        labelTop.pack(side=TOP)
        comboExs.pack(side=LEFT)
        # bBrowse.pack(side=LEFT)
        bPredictVdo.pack(side=LEFT)
        # bInputVdoData.pack(side=TOP)
        # bremoveJson.pack(side=BOTTOM)

    def framePredictLive(self, root):
        def preInputVdoData():
            try:
                owk = Owk.OpenWorkpout(root.filename, 'cam', None)
                owk._OpenCVpose()
            except Exception as e:
                print(e)

        Prframelive = Frame(root, bd="3", relief=GROOVE, padx=10, pady=10, bg='snow')
        cen = Label(Prframelive, text="Import Data", bg='snow')
        bBrowse = Button(Prframelive, text=' 1 - Browse ', bd=3, font=('', 10), padx=5, pady=5, command=self.selection)
        bInputVdoData = Button(Prframelive, text=" 2 - import Data", command=preInputVdoData)
        bremoveJson = Button(Prframelive, text="clear data", command=removeJson)
        pathlabel = Label(Prframelive, text=" -------------------------------------------", bg='snow')
        pathlabel2 = Label(Prframelive, text=" -------------------------------------------", bg='snow')
        pathlabel3 = Label(Prframelive, text=" -------------------------------------------", bg='snow')
        blive = Button(Prframelive, text=' live ', bd=3, font=('', 10), padx=20, pady=20, command=self.pageLive)
        cen.pack()
        pathlabel3.pack()
        bBrowse.pack(side=TOP)
        bInputVdoData.pack(side=TOP)
        pathlabel.pack()
        bremoveJson.pack(side=BOTTOM)
        pathlabel2.pack(side=BOTTOM)
        blive.pack(side=BOTTOM)
        Prframelive.pack(side=TOP)
        # blive.configure(text='Live Start...')

    def framePredictTypeWorkOutData(self, root):

        Prframe = Frame(root, bd="3", relief=GROOVE, padx=10, pady=10, bg='snow')
        Prframe.pack(side=TOP)
        label1 = Label(Prframe, text="Predict Type Workout", bg='snow').grid(row=0, column=0)
        label3 = Label(Prframe, text="   ", bg='snow').grid(row=2, column=0)
        bshow = Button(Prframe, text="k-nearest neighbors", command=predict_knn).grid(row=3, column=0)
        bshow2 = Button(Prframe, text="DecisionTree", command=predict_DecisionTree).grid(row=3, column=1)
        bshow3 = Button(Prframe, text="RandomForest", command=predict_RandomForest).grid(row=3, column=2)
        bshow4 = Button(Prframe, text="    SVC    ", command=predict_Svc).grid(row=3, column=3)
        bshow5 = Button(Prframe, text="MLPClassifier", command=predict_Mlpc).grid(row=3, column=4)

    def _createMenu(self):
        menubar = Menu(root)
        filemenu = Menu(menubar, tearoff=0)
        filemenu.add_command(label="New", command=self.donothing)
        filemenu.add_command(label="Train Model", command=self.input_Data)
        # filemenu.add_command(label="Predict_TypeExercise", command=self.framePredictVdo)
        # filemenu.add_command(label="Predict_Workout", command=self.framePredictVdo)
        # filemenu.add_command(label="Predict_ShowData", command=self.pageShow)
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
        root.filename = filedialog.askopenfilename(initialdir="/home/aditep/soflware/OpenWorkOut/vdo",
                                                   title="Select file",
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
        bshow = Button(fileshow, text="Knn", command=predict_knn)
        bshow2 = Button(fileshow, text="DecisionTree", command=predict_DecisionTree)
        bshow3 = Button(fileshow, text="RandomForest", command=predict_RandomForest)
        bshow4 = Button(fileshow, text="SVC", command=predict_Svc)
        bshow5 = Button(fileshow, text="MLPClassifier", command=predict_Mlpc)
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
            owk = Owk.OpenWorkpout(root.filename, comboExs.get(), None)
            owk._OpenCVpose()

        def InputLive():
            try:
                owk = Owk.OpenWorkpout(0, 'unknown', None)
                owk._OpenCVpose()
            except Exception as e:
                messagebox.showinfo("Error", e)
                print(e)

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
        blive = Button(Prframe, text=' live ', bd=3, font=('', 10), padx=20, pady=20, command=InputLive)
        label1 = Label(Prframe2, text="Training Model")

        # print(dict(comboExs))
        comboExs.pack()
        bBrowse.pack()
        label1.pack(side=TOP)
        binputData.pack()
        # blive.pack()
        btrainType.pack(side=LEFT)
        btrainWorkout.pack(side=RIGHT)


if __name__ == '__main__':
    root = tk.Tk()
    # content = Frame(root)
    MainPage(root)
    root.mainloop()  # Start GUI
