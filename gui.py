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
from PIL import ImageTk, Image

selectionnFilename = None


class MainPage():
    print('MainPage Ok...')

    def __init__(self, root):
        self.root = root
        self.pageMain()
        # self._createMenu()
        self.selectionnFilename = selectionnFilename

    def pageLive(self):
        try:
            owk = Owk.OpenWorkpout(0, 'cam', None, 'Live')
            owk._OpenCVpose()
        except Exception as e:
            messagebox.showinfo("Error", e)
            print(e)

    def pageMain(self):
        print('GUI Start...')
        root.title("OpenWorkOut by aditep campira v.1")
        root.geometry('800x330')
        root.resizable(0, 0)
        root.configure(background='#22242A')
        style = ttk.Style()

        style.theme_create("AditepStyle", parent="alt", settings={
            "TNotebook": {"configure": {"tabmargins": [2, 5, 2, 0]}},
            "TNotebook.Tab": {
                "configure": {"padding": [5, 1], "background": '#22242A'},
                "map": {"background": [("selected", '#f0b22a')],
                        "expand": [("selected", [1, 1, 1, 0])]}},
            "TFrame": {"configure": {"background": '#22242A'}}
        })

        style.theme_use("AditepStyle")
        self.frameTap(root)
        # self.framePredictLive(root)
        # self.framePredictVdo(root)
        # self.framePredictTypeWorkOutData(root)

    def frameTap(self, root):
        tab_control = ttk.Notebook(root)
        tab1 = ttk.Frame(tab_control)
        tab2 = ttk.Frame(tab_control)
        load_tab1 = Image.open('pic/gui/tabs1.png')
        render_load_tab1 = ImageTk.PhotoImage(load_tab1)
        load_tab2 = Image.open('pic/gui/tabs2.png')
        render_load_tab2 = ImageTk.PhotoImage(load_tab2)
        tab_control.add(tab1, text='Predict Exercise', image=render_load_tab1)
        tab_control.add(tab2, text='Predict Workout', image=render_load_tab2)
        tab_control.image = render_load_tab2, render_load_tab1

        self.framePredictLive(tab1, root)
        self.framePredictVdo(tab2, root)
        tab_control.pack(expand=1, fill='both')
        frameTap = Frame(root, bd=10, relief=FLAT, bg='#DD6161')

        load_rainmodel = Image.open('pic/gui/trainmodel.png')
        render_load_rainmodel = ImageTk.PhotoImage(load_rainmodel)
        bInputVdoData = Button(root, image=render_load_rainmodel, relief=FLAT, command=self.input_Data)
        bInputVdoData.image = render_load_rainmodel
        bInputVdoData.pack()
        frameTap.pack(fill=Y, side=LEFT)

    def framePredictVdo(self, tabs, root):
        def preInputLiveData():
            removeJson()
            label1.configure(text=comboExs.get())
            print(comboExs.get())
            owk = Owk.OpenWorkpout(0, 'predictVdo', comboExs.get(), 'Predict Workout Live')
            owk._OpenCVpose()

        def preInputVdoData():

            try:
                self.selection()
                label1.configure(text=comboExs.get())
                print(comboExs.get())
                owk = Owk.OpenWorkpout(root.filename, 'predictVdo', comboExs.get(), 'Predict Workout VDO')
                owk._OpenCVpose()
            except AttributeError:
                messagebox.showinfo('Upload File VDO', "Upload VDO ")

        PrframeVdo = Frame(tabs, bd=10, relief=FLAT, padx=42.5, pady=10, bg='#353F53')
        PrframeVdo2 = Frame(PrframeVdo, relief=FLAT, padx=42.5, pady=10, bg='#353F53')
        PrframeVdo.pack(fill=X, expand=True, side=TOP)
        load_bpredictLive = Image.open('pic/gui/b_predictLive.png')
        render_load_bpredictLive = ImageTk.PhotoImage(load_bpredictLive)
        load_bpredictVDO = Image.open('pic/gui/b_predictVDO.png')
        render_load_bpredictVDO = ImageTk.PhotoImage(load_bpredictVDO)
        load_Lpredictworkout = Image.open('pic/gui/label_predictWorkout.png')
        render_load_Lpredictworkou = ImageTk.PhotoImage(load_Lpredictworkout)
        load_label_traing_live = Image.open('pic/gui/label_traing_live.png')
        render_load_label_traing_live = ImageTk.PhotoImage(load_label_traing_live)

        labelline = tk.Label(PrframeVdo, bd=10, bg='#353F53', image=render_load_label_traing_live)
        labelline.image = render_load_label_traing_live

        labelTop = tk.Label(PrframeVdo, bd=10, text="Choose your exercise", bg='#444953', font='Times 10 bold',
                            image=render_load_Lpredictworkou)
        labelTop.image = render_load_Lpredictworkou

        value = [
            "Push Ups",
            "Squat",
            "Deadlift",
            "Dumbbell Shoulder Press",
            "Barbell Curl"]

        comboExs = ttk.Combobox(PrframeVdo2, values=value)
        comboExs.current(1)

        bPredictVdo = Button(PrframeVdo, relief=FLAT,text="Predict VDO", command=preInputVdoData, image=render_load_bpredictVDO)
        bPredictLive = Button(PrframeVdo, relief=FLAT,text="Predict Live", command=preInputLiveData, image=render_load_bpredictLive)
        bPredictLive.image = render_load_bpredictLive
        bPredictVdo.image = render_load_bpredictVDO

        label1 = Label(PrframeVdo, text="")

        comboExs.pack(side=LEFT)
        labelline.pack(side=TOP)
        labelTop.pack(side=TOP)
        bPredictVdo.pack(side=LEFT)
        bPredictLive.pack(side=RIGHT)
        PrframeVdo2.pack()

    def framePredictLive(self, tabs, root):
        def preInputVdoData():
            try:
                removeJson()
                self.selection()
                owk = Owk.OpenWorkpout(root.filename, 'cam', None, 'Import Vdo')
                owk._OpenCVpose()
            except Exception as e:
                print(e)

        def predict():
            # names = predict_Mlpc()
            names = predict_knn()
            messagebox.showinfo('Exercise', 'Predict Exercise posture :: " ' + str(names).upper() + ' "')
            self.showpic(names)

        def remove_Json():
            try:
                MsgBox = tk.messagebox.askquestion('remove data',
                                                   'Are you sure you want clear data', icon='warning')
                if MsgBox == 'yes':
                    removeJson()
                    tk.messagebox.showinfo('Return', 'Clear Data finished')
                else:
                    tk.messagebox.showinfo('Return', 'You will now return to the application screen')
            except Exception as e:
                print(e)

        Prframelive = Frame(tabs, relief=FLAT, bg='#353F53')
        Prframe_uplode = Frame(Prframelive, relief=RIDGE, bg='#444953')
        Prframe_live = Frame(Prframe_uplode, relief=RIDGE, padx=10, pady=10, bg='#444953')
        Prframe_vdo = Frame(Prframe_uplode, relief=RIDGE, padx=10, pady=10, bg='#444953')
        Prframe_pre = Frame(Prframelive, relief=RIDGE, padx=10, pady=10, bg='#22242A')

        load_bupload = Image.open('pic/gui/b_upload.png')
        render_load_bupload = ImageTk.PhotoImage(load_bupload)
        load_brecord = Image.open('pic/gui/b_record.png')
        render_load_brecord = ImageTk.PhotoImage(load_brecord)
        load_bpredict = Image.open('pic/gui/b_predict.png')
        render_load_bpredict = ImageTk.PhotoImage(load_bpredict)
        load_Lpredict = Image.open('pic/gui/label_predictType.png')
        render_load_Lpredict = ImageTk.PhotoImage(load_Lpredict)

        cen = Label(Prframe_uplode, image=render_load_Lpredict)
        cen.image = render_load_Lpredict
        bBrowse = Button(Prframelive, text=' 1 - Browse ', bd=3, font=('', 10), padx=5, pady=5, command=self.selection)
        bInputVdoData = Button(Prframe_vdo, text=" Upload File VDO", command=preInputVdoData,
                               relief=FLAT, image=render_load_bupload, bg='#444953')
        bInputVdoData.image = render_load_bupload
        bshowpredctWorkout = Button(Prframe_pre, text="Predict Exercise posture", command=predict, bg='#22242A',
                                    image=render_load_bpredict)
        bshowpredctWorkout.image = render_load_bpredict

        bremoveJson = Button(Prframe_uplode, text="clear data", command=remove_Json, bg='red')
        pathlabel1 = Label(Prframe_uplode, text=" Predict Exercise posture", bg='#444953', font='Times 10 bold',
                           fg='snow')
        blive = Button(Prframe_live, relief=FLAT, command=self.pageLive, image=render_load_brecord, bg='#444953')
        blive.image = render_load_brecord

        Prframelive.pack(fill=X, expand=True)
        Prframe_uplode.pack(side=LEFT)
        # pathlabel1.pack(side=TOP)
        cen.pack(side=TOP)
        Prframe_vdo.pack(side=LEFT)
        Prframe_live.pack(side=LEFT)

        Prframe_pre.pack(fill=X)

        # pathlabel1.pack()
        # bBrowse.pack(side=TOP)
        bInputVdoData.pack(side=TOP)
        blive.pack()
        # bremoveJson.pack(side=BOTTOM)
        bshowpredctWorkout.pack(side=BOTTOM)

        # blive.configure(text='Live Start...')

    def framePredictTypeWorkOutData(self, root):

        Prframe = Frame(root, bd="3", relief=GROOVE, padx=10, pady=10, bg='#444953')
        Prframe.pack(side=TOP)
        label1 = Label(Prframe, text="Predict Type Workout", bg='snow').grid(row=0, column=0)
        label3 = Label(Prframe, text="   ", bg='snow').grid(row=2, column=0)
        bshow = Button(Prframe, text="k-nearest neighbors", command=predict_knn).grid(row=3, column=0)
        bshow2 = Button(Prframe, text="DecisionTree", command=predict_DecisionTree).grid(row=3, column=1)
        bshow3 = Button(Prframe, text="RandomForest", command=predict_RandomForest).grid(row=3, column=2)
        bshow4 = Button(Prframe, text="    SVC    ", command=predict_Svc).grid(row=3, column=3)
        bshow5 = Button(Prframe, text="MLPClassifier", command=predict_Mlpc).grid(row=3, column=4)

    def _createMenu(self):
        menubar = Menu(root, bg='#34383c', fg='snow')
        filemenu = Menu(menubar, tearoff=0)
        # filemenu.add_command(label="New", command=self.donothing)
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

    def input_Data(self):
        print('input_Data')

        def inputData():
            try:
                MsgBox = tk.messagebox.askquestion('Import File',
                                                   'Are you sure you want import data for....   " ' + comboExs.get() + ' "',
                                                   icon='warning')
                if MsgBox == 'yes':
                    self.selection()
                    label1.configure(text=comboExs.get())
                    print(comboExs.get())
                    owk = Owk.OpenWorkpout(root.filename, comboExs.get(), None, 'Import Data Vdo')
                    owk._OpenCVpose()

            except UnboundLocalError as e:
                messagebox.showinfo("Error", 'Import File VDO')

        def InputLive():
            try:
                MsgBox = tk.messagebox.askquestion('Import File',
                                                   'Are you sure you want import data for....   " ' + comboExs.get() + ' "',
                                                   icon='warning')
                if MsgBox == 'yes':
                    owk = Owk.OpenWorkpout(0, 'unknown', None, 'Import Data Live')
                    owk._OpenCVpose()
            except Exception as e:
                messagebox.showinfo("Error", e)
                print(e)

        def trainType():
            messagebox.showinfo("Traning", 'Training Model Start')
            training_knn()
            # training_Svc()
            # training_RandomForest()
            # training_percep()
            training_mlpc()
            # training_DecisionTree()
            messagebox.showinfo("Traning", 'Training Model finished')

        def trainingEx():
            MsgBox = tk.messagebox.askquestion('Import File',
                                               'Are you sure you wan Training Model " ' + comboExs.get() + ' "',
                                               icon='question')
            if MsgBox == 'yes':
                # training_knnEx(comboExs.get())
                # training_DecisionTreeEx(comboExs.get())
                # training_SvcEx(comboExs.get())
                # training_RandomForestEx(comboExs.get())
                training_mlpcEx(comboExs.get())
                messagebox.showinfo("Traning", 'Training Model ' + comboExs.get() + ' finished')

        fileTrain = Toplevel(root)
        fileTrain.geometry('800x400')
        fileTrain.title("Training Model/Input Data")
        fileTrain.resizable(0, 0)
        fileTrain.configure(background='#22242A')
        # pwTrain1 = PanedWindow(fileTrain, bg='red', orient=VERTICAL)
        # pwTrain1.pack(fill=BOTH, expand=1)

        load_b_train_typeEx = Image.open('pic/gui/b_train_typeEx.png')
        render_load_b_train_typeEx = ImageTk.PhotoImage(load_b_train_typeEx)
        load_b_train_workout = Image.open('pic/gui/b_train_workout.png')
        render_load_b_train_workout = ImageTk.PhotoImage(load_b_train_workout)
        load_label_traing = Image.open('pic/gui/label_traing.png')
        render_load_label_traing = ImageTk.PhotoImage(load_label_traing)
        load_label_traing_live = Image.open('pic/gui/label_traing_live.png')
        render_load_label_traing_live = ImageTk.PhotoImage(load_label_traing_live)
        load_brecord = Image.open('pic/gui/b_record.png')
        render_load_brecord = ImageTk.PhotoImage(load_brecord)

        Prframe = Frame(fileTrain, bd="3", relief=FLAT, padx=100, pady=50, bg='snow')
        Prframe.pack(side=TOP)
        Prframe2 = Frame(fileTrain, bd="3", relief=FLAT, padx=20, pady=20, bg='snow')
        Prframe2.pack()
        labelTop = tk.Label(Prframe,
                            text="Choose your exercise for inputData and Train_workout", bg='snow',
                            image=render_load_label_traing)
        labelTop.image = render_load_label_traing
        labelTop.pack()
        value = [
            "Push Ups",
            "Squat",
            "Deadlift",
            "Dumbbell Shoulder Press",
            "Barbell Curl",
            "unknown"]
        load_b_upload = Image.open('pic/gui/b_upload.png')
        render_load_b_upload = ImageTk.PhotoImage(load_b_upload)

        comboExs = ttk.Combobox(Prframe, values=value)
        comboExs.current(1)
        binputData = Button(Prframe, text="Input Data", command=inputData, image=render_load_b_upload)
        binputData.image = render_load_b_upload
        btrainType = Button(Prframe2, text="Train_Type", command=trainType, image=render_load_b_train_typeEx)
        btrainType.image = render_load_b_train_typeEx
        btrainWorkout = Button(Prframe2, text="Train_workout", command=trainingEx, image=render_load_b_train_workout)
        btrainWorkout.image = render_load_b_train_workout
        bBrowse = Button(Prframe, text=' Browse ', bd=3, font=('', 10), padx=5, pady=5, command=self.selection)
        pathlabel2 = Label(Prframe, text=" -----------------------Live-----------------------", bg='snow',
                           image=render_load_label_traing_live)
        pathlabel2.image = render_load_label_traing_live
        blive = Button(Prframe, text=' live ', bd=3, font=('', 10), padx=20, pady=20, command=InputLive,
                       image=render_load_brecord)
        blive.image = render_load_brecord
        label1 = Label(Prframe2, text="Training Model", bg='snow')

        # print(dict(comboExs))
        comboExs.pack()
        # bBrowse.pack()
        # label1.pack(side=TOP)
        pathlabel2.pack()
        binputData.pack(side=LEFT, expand=True)
        blive.pack(side=LEFT, expand=True)
        btrainType.pack(side=LEFT)
        btrainWorkout.pack(side=RIGHT)

    def showpic(self, namepic):
        showpic = Toplevel(root)
        showpic.geometry('450x400')
        showpic.title("Show Pic")
        pwshow = PanedWindow(showpic, bg='#444953', orient=VERTICAL)
        pwshow.pack(fill=BOTH, expand=1)
        load = Image.open('pic/' + namepic + '.jpg')
        render = ImageTk.PhotoImage(load)
        img = Label(pwshow, image=render)
        img.image = render
        img.place(x=0, y=0)


if __name__ == '__main__':
    root = tk.Tk()
    # content = Frame(root)
    MainPage(root)

    root.mainloop()  # Start GUI
