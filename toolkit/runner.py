import tkinter.filedialog
from tkinter import *
from tkinter import ttk
#import numpy as np
#from PIL import Image,ImageTk
import kan_train_tf,sentiment_analysis_movie_batch_inference,sentiment_analysis_movie_batch,spam_not_spam_train,sapm_not_spam_inference
import sequence_prediction_lstm_train,sequence_prediction_lstm_inference
import threading,time,json
import do_inference,os

ALGOS=[("CNN",1),("RNN",2),("Word Embeddings",3)]
var,rb=None,None
pathLabel=None
filename="C:\\MLDatabases\\kannadaHandWrittenImages"
imagesLabels=[]
weightLabels=[]
progressLabel=None,None
epoch_label,steps_label,train_accuracy_label,val_accuracy_label,test_accuracy_label=[None]*5
hyperParameterFrame=None
train_page=None
CNNhyperParameterFrame=None
RNNhyperParameterFrame=None
strides=tuple()
filters=tuple()
lr=0.001
num_class=None
l1h,l1w,l1s,l2h,l2w,l2s,l3h,l3w,l3s,l4h,l4w,l4s,num_class=[None]*13
lr,rnn_lr=None,None
input_channel=3
stridelistbox1,stridelistbox2,stridelistbox3,stridelistbox4=[None]*4
fc1,fc2,dropout,same,valid=[None]*5

val_image__selection_frame,valimselbutton=None,None
rnn_lr,rnn_dr,epo_entry=None,None,None


stride_options=[[1,1,1,1],[1,2,2,1],[1,3,3,1],[1,4,4,1]]
imageFrame,progressFrame=None,None

renn_usecase_var=None

"""

VALIDATION

"""

selected_model_name=""
model_name_label=None
val_im=""
valim_name_label=None
model_selection_frame=None
val_result_label=""
val_drop=None
val_lr=0.001
val_container_frame=None
val_frame=None
sentence = None
"""
def updater(q,imagesLabels,weightLabels):
    
    while ~q.empty():
        q_item=q.get(block=True)
        imagesLabels=[l.config(image=img) for l,img in zip(imagesLabels,q_item[0])]
        weightLabels=[wl.config(image=img) for wl,img in zip(weightLabels,q_item[1])]
"""

def radio_button_cmd():
    global var,val_image__selection_frame,valimselbutton
    i=int(var.get())
    
    if i==1:
        build_cnn_hp_frame()
        val_image__selection_frame.config(text="Image Selection")
        valimselbutton.config(text="Select Image")       
        
    elif i==2:
        get_rnn_hp_frame()
        #val_image__selection_frame.config(text="Select input")
        #valimselbutton.config(text="Select txt file")
        build_rnn_validation_frame()
    else:
        pass

def build_rnn_validation_frame():
    global val_frame,val_container_frame,model_name_label,sentence,val_result_label
    
    val_frame.destroy()
    val_frame=ttk.Frame(master=val_container_frame)
    val_frame.pack(side=TOP,expand=True,fill="both")

    model_selection_frame=ttk.LabelFrame(master=val_frame,text="Select Model",style="Gen.TLabelframe")
    model_selection_frame.pack(side=TOP,expand=False,fill="both")
    modselbutton=ttk.Button(master=model_selection_frame,text="Select model file",command=rnn_model_selection_button_press,style="Generic.TButton")
    modselbutton.pack(side=LEFT,anchor=N,expand=False,fill="y")
    model_name_label=ttk.Label(master=model_selection_frame,text="")
    model_name_label.pack(side=LEFT,anchor="center",fill="x",expand=True)
    
    
    val_image__selection_frame=ttk.LabelFrame(master=val_frame,text="Enter Sentence",style="Gen.TLabelframe")
    val_image__selection_frame.pack(side=TOP,expand=False,fill="both")
    sentence = Text ( master=val_frame)
    sentence.pack(side=TOP,expand=False,fill="both")
    #valimselbutton=ttk.Button(master=val_image__selection_frame,text="Select Image file",command=valimselbutton_press,style="Generic.TButton")
    #valimselbutton.pack(side=LEFT,anchor=N,expand=False,fill="y")
    valim_name_label=ttk.Label(master=val_image__selection_frame,text="")
    valim_name_label.pack(side=LEFT,anchor="center",fill="x",expand=True)
    
    """
    param_frame=ttk.LabelFrame(master=val_frame,text="Parameters",style="Gen.TLabelframe")
    param_frame.pack(side=TOP,expand=False,fill="both")
    
    val_lr_label=ttk.Label(master=param_frame,text="Learning Rate: ",font=('courier',12,'bold'))
    val_lr_label.pack(side=LEFT,expand=True,fill="both")
    
    val_lr=Entry(master=param_frame,bd=5,font=('courier',12,'bold'))
    val_lr.pack(side=LEFT,expand=True,fill="both")
    
    val_drop_label=ttk.Label(master=param_frame,text="Dropout Rate: ",font=('courier',12,'bold'))
    val_drop_label.pack(side=LEFT,expand=True,fill="both")
    
    val_drop=Entry(master=param_frame,bd=5,font=('courier',12,'bold'))
    val_drop.pack(side=LEFT,expand=True,fill="both")
    """
    inf_frame=ttk.Frame(master=val_frame)
    inf_frame.pack(side=TOP,expand=False,fill="both")
    inf_button=ttk.Button(master=inf_frame,text="Run Inference",command=run_stateless_rnn_inference,style="Generic.TButton")
    inf_button.pack(side=LEFT,anchor="center",expand=True,fill="y")
        
    val_result_frame=ttk.LabelFrame(master=val_frame,text="Result",style="Gen.TLabelframe")
    val_result_frame.pack(side=TOP,expand=True,fill="both")
    val_result_label=ttk.Label(master=val_result_frame,text="",font=("Calibri", 20))
    val_result_label.pack(side=LEFT,anchor="center",fill="x",expand=True)
 
def rnn_model_selection_button_press():
    global selected_model_name,model_name_label
    selected_model_name=filedialog.askopenfilename(initialdir = "D:\\Deep Learning\\model\\RNN\\sentiment",title = "Select model file",filetypes = (("meta","*.meta"),("all files","*.*")))
    model_name_label.config(text=selected_model_name)
    
def run_stateless_rnn_inference():
    global val_result_label,sentence,selected_model_name
    content=sentence.get(1.0, END)
    
    rnn_case=get_rnn_use_case()
    
    
    if rnn_case==2:
        k = threading.Thread(target=sentiment_analysis_movie_batch_inference.run,args=(content,val_result_label,selected_model_name))
        k.daemon=True
        k.start()
    elif rnn_case==1:
        
        k = threading.Thread(target=sapm_not_spam_inference.run,args=(content,val_result_label,selected_model_name))
        k.daemon=True
        k.start()
    elif rnn_case==3:
        k = threading.Thread(target=sequence_prediction_lstm_inference.run,args=(content,val_result_label,selected_model_name))
        k.daemon=True
        k.start()
                
        
    else:
        pass
    

def browse_button_press():
    
    global var,filename    
    
    i=int(var.get())
    
    if i==1:
    
        filename=filedialog.askdirectory()
        print(filename)
        pathLabel.config(text=filename)
        #stridelistbox1.selection_set(0)
    elif i==2:
        filename=filedialog.askdirectory()
        print(filename)
        pathLabel.config(text=filename)
        
    else:     
        pass

def get_hyperParameterFrame():
    
    s = ttk.Style()
    s.configure('Red.TLabelframe.Label', font=('courier', 15, 'bold'))
    
    global hyperParameterFrame,CNNhyperParameterFrame,RNNhyperParameterFrame
    
    hyperParameterFrame=ttk.LabelFrame(master=train_page,text="Select hyperparameters",style = "Red.TLabelframe")
    hyperParameterFrame.pack(side=TOP,expand=True,fill="both")
            
    CNNhyperParameterFrame=ttk.Frame(master=hyperParameterFrame)
    CNNhyperParameterFrame.pack(side=TOP,expand=True,fill="both")
            
    RNNhyperParameterFrame=ttk.Frame(master=hyperParameterFrame)
    RNNhyperParameterFrame.pack(side=TOP,expand=True,fill="both")
    
    return hyperParameterFrame


def load_press():
    global hyperParameterFrame,train_page,RNNhyperParameterFrame,CNNhyperParameterFrame
    global l1h,l1w,l1s,l2h,l2w,l2s,l3h,l3w,l3s,l4h,l4w,l4s
    global stridelistbox
    global lr,rnn_lr,rb
    global fc1,fc2,dropout,same,valid,num_class
    global stridelistbox1,stridelistbox2,stridelistbox3,stridelistbox4 
    
    
    load_file=filedialog.askopenfilename(initialdir = "D:\\DL\\KannadaData\\lessData\\param",title = "Select file",filetypes = ((".json","*.json"),("all files","*.*")))
    
    with open(load_file,"r") as f:
        data=json.load(f)
    
    stridelistbox1.selection_clear(0, END)
    stridelistbox2.selection_clear(0, END)
    stridelistbox3.selection_clear(0, END)
    stridelistbox4.selection_clear(0, END)
    
    stridelistbox1.selection_set(data["s1"][0])
    stridelistbox2.selection_set(data["s2"][0])
    stridelistbox3.selection_set(data["s3"][0])
    stridelistbox4.selection_set(data["s4"][0])
    
    l1h.delete(0,"end")
    l1h.insert(0,data["l1h"])
    
    l1w.delete(0,"end")
    l1w.insert(0,data["l1w"])

    l1s.delete(0,"end")
    l1s.insert(0,data["l1s"])

    l2h.delete(0,"end")
    l2h.insert(0,data["l2h"])

    l2w.delete(0,"end")
    l2w.insert(0,data["l2w"])

    l2s.delete(0,"end")
    l2s.insert(0,data["l2s"])

    l3h.delete(0,"end")
    l3h.insert(0,data["l3h"])

    l3w.delete(0,"end")
    l3w.insert(0,data["l3h"])

    l3s.delete(0,"end")
    l3s.insert(0,data["l3s"])

    l4h.delete(0,"end")
    l4h.insert(0,data["l4h"])

    l4w.delete(0,"end")
    l4w.insert(0,data["l4w"])

    l4s.delete(0,"end")
    l4s.insert(0,data["l4s"])
    
    fc1.delete(0,"end")
    fc1.insert(0,data["fcl1"])
    fc2.delete(0,"end")
    fc2.insert(0,data["fcl2"])
    
    lr.delete(0,"end")
    lr.insert(0,data["lr"])   
    dropout
    
    dropout.delete(0,"end")
    dropout.insert(0,data["drop_rate"])
    
    rb.set(1 if data["padding"]=="SAME" else 2)
    
    num_class.delete(0,"end")
    num_class.insert(0,data["num_class"])
    
def save_press():
    global hyperParameterFrame,train_page,RNNhyperParameterFrame,CNNhyperParameterFrame
    global l1h,l1w,l1s,l2h,l2w,l2s,l3h,l3w,l3s,l4h,l4w,l4s
    global stridelistbox
    global lr,rnn_lr,rb
    global fc1,fc2,dropout,same,valid,num_class
    global stridelistbox1,stridelistbox2,stridelistbox3,stridelistbox4 
    
    f = filedialog.asksaveasfile(mode='w', defaultextension=".json")
    if f is None: # asksaveasfile return `None` if dialog closed with "cancel".
        return
    
    sel_idx1=stridelistbox1.curselection()
    sel_idx2=stridelistbox2.curselection()
    sel_idx3=stridelistbox3.curselection()
    sel_idx4=stridelistbox4.curselection()
    
    if is_empty(sel_idx1):
        sel_idx1=(0,)
    if is_empty(sel_idx2):
        sel_idx2=(0,)
    if is_empty(sel_idx3):
        sel_idx3=(0,)
    if is_empty(sel_idx4):
        sel_idx4=(0,)
    lrv=lr.get()
    
    if is_empty(lrv):
        lrv=0.001

    drop_rate=str(dropout.get()).strip()
    if is_empty(drop_rate):
        drop_rate=0.001

    num_class_value=int(num_class.get())
    if is_empty(num_class_value):
        num_class_value=5
    
    padding= "SAME" if int(rb.get())==1 else "VALID"    
    data={"l1h":int(l1h.get()),"l1w":int(l1w.get()),"l1s":int(l1s.get()),"l2h":int(l2h.get()),"l2w":int(l2w.get()),"l2s":int(l2s.get()),"l3h":int(l3h.get()),"l3w":int(l3w.get()),"l3s":int(l3s.get()),"l4h":int(l4h.get()),"l4w":int(l4w.get()),"l4s":int(l4s.get()),"s1":sel_idx1,"s2":sel_idx2,"s3":sel_idx3,"s4":sel_idx4,"lr":lrv,"drop_rate":drop_rate,"padding":padding,"fcl1":int(fc1.get()),"fcl2":int(fc2.get()),"num_class":num_class_value}

    
    json.dump(data,f)
    f.close()


def build_cnn_hp_frame():
        
        global hyperParameterFrame,train_page,RNNhyperParameterFrame,CNNhyperParameterFrame
        global l1h,l1w,l1s,l2h,l2w,l2s,l3h,l3w,l3s,l4h,l4w,l4s
        global stridelistbox
        global lr,rnn_lr,rb
        global fc1,fc2,dropout,same,valid,num_class
        global stridelistbox1,stridelistbox2,stridelistbox3,stridelistbox4
        
        global imageFrame,  progressFrame
        
        rb=IntVar()
        
        RNNhyperParameterFrame.destroy()
        CNNhyperParameterFrame.destroy()
        #hyperParameterFrame=ttk.LabelFrame(master=train_page,text="Select hyperparameters")
        #hyperParameterFrame.pack(side=TOP,expand=True,fill="both")
        
        subStyle=ttk.Style()
        subStyle.configure('Red1.TLabelframe.Label', font=('courier', 12, 'bold'))
        
        CNNhyperParameterFrame=ttk.Frame(master=hyperParameterFrame)
        CNNhyperParameterFrame.pack(side=TOP,expand=True,fill="both")
        strideFrame=ttk.LabelFrame(master=CNNhyperParameterFrame,text="Strides(Columns=Layers)",style="Red1.TLabelframe")
        strideFrame.pack(side=LEFT,expand=True,fill="both")
        filterframe=ttk.LabelFrame(master=CNNhyperParameterFrame,text="Filter(rows=Layers)",style="Red1.TLabelframe")
        filterframe.pack(side=LEFT,expand=True,fill="both")
        misc=ttk.LabelFrame(master=CNNhyperParameterFrame,text="Misc",style="Red1.TLabelframe")
        misc.pack(side=LEFT,expand=True,fill="both")


        strideFrame1=ttk.Frame(master=strideFrame)
        strideFrame1.pack(side=LEFT,expand=True,fill="both")
        
        stridelistbox1 = Listbox(master=strideFrame1,exportselection=False,selectmode=SINGLE,height=5,width=10,font=('courier', 12, 'bold'))
        
        strideFrame2=ttk.Frame(master=strideFrame)
        strideFrame2.pack(side=LEFT,expand=True,fill="both")
        
        stridelistbox2 = Listbox(master=strideFrame2,exportselection=False,height=5,selectmode=SINGLE,width=10,font=('courier', 12, 'bold'))
        
        strideFrame3=ttk.Frame(master=strideFrame)
        strideFrame3.pack(side=LEFT,expand=True,fill="both")
        
        stridelistbox3 = Listbox(master=strideFrame3,exportselection=False,height=5,selectmode=SINGLE,width=10,font=('courier', 12, 'bold'))
        
        strideFrame4=ttk.Frame(master=strideFrame)
        strideFrame4.pack(side=LEFT,expand=True,fill="both")
        
        stridelistbox4 = Listbox(master=strideFrame4,exportselection=False,height=5,selectmode=SINGLE,width=10,font=('courier', 12, 'bold'))
        
        for item in ["1","2","3","4"]:
            stridelistbox1.insert(END,item)
        stridelistbox1.pack(side=LEFT,expand=True,fill="both")
        
        for item in ["1","2","3","4"]:
            stridelistbox2.insert(END,item)
        stridelistbox2.pack(side=LEFT,expand=True,fill="both")
        
        for item in ["1","2","3","4"]:
            stridelistbox3.insert(END,item)     
        stridelistbox3.pack(side=LEFT,expand=True,fill="both")
        
        for item in ["1","2","3","4"]:
            stridelistbox4.insert(END,item)        
        stridelistbox4.pack(side=LEFT,expand=True,fill="both")
        
        layer1_filter=ttk.Frame(master=filterframe)    
        l1h = Spinbox(master=layer1_filter, from_=5, to=10,width=10,font=('courier', 12, 'bold'))
        l1h.pack(side=LEFT,expand=True,fill="both")
        l1w = Spinbox(layer1_filter, from_=5, to=10,width=10,font=('courier', 12, 'bold'))
        l1w.pack(side=LEFT,expand=True,fill="both")
        l1s = Spinbox(layer1_filter, from_=32, to=256,increment=32,width=10,font=('courier', 12, 'bold'))
        l1s.pack(side=LEFT,expand=True,fill="both")
        layer1_filter.pack(side=TOP,expand=True,fill="both")
        
        
        layer2_filter=ttk.Frame(master=filterframe)
        layer2_filter.pack(side=TOP,expand=True,fill="both")
        l2h = Spinbox(layer2_filter, from_=5, to=10,width=10,font=('courier', 12, 'bold'))
        l2h.pack(side=LEFT,expand=True,fill="both")
        l2w = Spinbox(layer2_filter, from_=5, to=10,width=10,font=('courier', 12, 'bold'))
        l2w.pack(side=LEFT,expand=True,fill="both")
        l2s = Spinbox(layer2_filter, from_=32, to=256,increment=32,width=10,font=('courier', 12, 'bold'))
        l2s.pack(side=LEFT,expand=True,fill="both")
        
        layer3_filter=ttk.Frame(master=filterframe)
        l3h = Spinbox(layer3_filter, from_=5, to=10,width=10,font=('courier', 12, 'bold'))
        l3h.pack(side=LEFT,expand=True,fill="both")
        l3w = Spinbox(layer3_filter, from_=5, to=10,width=10,font=('courier', 12, 'bold'))
        l3w.pack(side=LEFT,expand=True,fill="both")
        l3s = Spinbox(layer3_filter, from_=32, to=256,increment=32,width=10,font=('courier', 12, 'bold'))
        l3s.pack(side=LEFT,expand=True,fill="both")
        layer3_filter.pack(side=TOP,expand=True,fill="both")
        

        layer4_filter=ttk.Frame(master=filterframe)
        l4h = Spinbox(layer4_filter, from_=5, to=10,width=10,font=('courier', 12, 'bold'))
        l4h.pack(side=LEFT,expand=True,fill="both")
        l4w = Spinbox(layer4_filter, from_=5, to=10,width=10,font=('courier', 12, 'bold'))
        l4w.pack(side=LEFT,expand=True,fill="both")
        l4s = Spinbox(layer4_filter, from_=32, to=256,increment=32,width=10,font=('courier', 12, 'bold'))
        l4s.pack(side=LEFT,expand=True,fill="both")
        layer4_filter.pack(side=TOP,expand=True,fill="both")
        
        
        lr_frame=ttk.Frame(master=misc)
        lr_frame.pack(anchor=W,side=TOP,expand=False,fill=None)
        
        lr_label=ttk.Label(master=lr_frame,text="Learning Rate: ",font=('courier', 12, 'bold'))
        lr_label.pack(side=LEFT,expand=False,fill=None)
        
        lr=Entry(master=lr_frame,bd=5)
        lr.pack(side=LEFT,expand=False,fill=None)
        
        dropout_frame=ttk.Frame(master=misc)
        dropout_frame.pack(anchor=W,side=TOP,expand=False,fill=None)
        
        dropout_label=ttk.Label(master=dropout_frame,text="Dropout Rate: ",font=('courier', 12, 'bold'))
        dropout_label.pack(side=LEFT,expand=False,fill=None)
        
        dropout=Entry(master=dropout_frame,bd=5)
        dropout.pack(side=LEFT,expand=False,fill=None)
        
        fc_frame=ttk.Frame(master=misc)
        fc_frame.pack(anchor=W,side=TOP,expand=False,fill=None)
        fc1_l=ttk.Label(master=fc_frame,text="FC1 H units",font=('courier', 12, 'bold'))
        fc1_l.pack(side=LEFT,expand=False,fill=None)        
        fc1 = Spinbox(fc_frame, from_=0, to=2000,increment=100,font=('courier', 12, 'bold'))
        fc1.pack(side=LEFT,expand=True,fill="both")
        fc2_l=ttk.Label(master=fc_frame,text="Out H units",font=('courier', 12, 'bold'))
        fc2_l.pack(side=LEFT,expand=False,fill=None)
        fc2 = Spinbox(fc_frame, from_=0, to=2000,increment=100,font=('courier', 12, 'bold'))
        fc2.pack(side=LEFT,expand=True,fill="both")
        
        padding_frame=ttk.Frame(master=misc)
        padding_frame.pack(anchor=W,side=TOP,expand=False,fill=None)
        padding_l=ttk.Label(master=padding_frame,text="Padding: ",font=('courier', 12, 'bold'))
        padding_l.pack(side=LEFT,expand=False,fill=None)
        
        same=ttk.Radiobutton(master=padding_frame,value=1,variable=rb,text="SAME")
        same.pack(side=LEFT,anchor=N,expand=True,fill="both")
        valid=ttk.Radiobutton(master=padding_frame,value=2,variable=rb,text="VALID")
        valid.pack(side=LEFT,anchor=N,expand=True,fill="both")
        
        num_class_frame=ttk.Frame(master=misc)
        num_class_frame.pack(anchor=W,side=TOP,expand=False,fill=None)
        num_class_l=ttk.Label(master=num_class_frame,text="Num Class: ",font=('courier', 12, 'bold'))
        num_class_l.pack(side=LEFT,expand=False,fill=None)
        num_class = Spinbox(num_class_frame, from_=0, to=1000,increment=1,font=('courier', 12, 'bold'))
        num_class.pack(side=LEFT,expand=True,fill="both")
        
        load_save_frame=ttk.Frame(master=misc)
        load_save_frame.pack(anchor=W,side=TOP,expand=False,fill=None)
        
        load_button=ttk.Button(master=load_save_frame,text="Load",command=load_press)
        load_button.pack(anchor=W,side=LEFT,expand=False,fill=None)
        save_button=ttk.Button(master=load_save_frame,text="Save",command=save_press)
        save_button.pack(anchor=W,side=LEFT,expand=False,fill=None)
        
        imageFrame.config(text="ACTIVATION MAP Visualization")
        progressFrame.config(text="WEIGHT Visualization")
        

def get_rnn_use_case():
    global renn_usecase_var    
    case=int(renn_usecase_var.get())
    return case
           
        
def get_rnn_hp_frame():
    
        global hyperParameterFrame,RNNhyperParameterFrame,CNNhyperParameterFrame,imageFrame,progressFrame
        global rnn_lr,rnn_dr,epo_entry,renn_usecase_var
        
        RNNhyperParameterFrame.destroy()
        CNNhyperParameterFrame.destroy()
        
        subStyle=ttk.Style()
        subStyle.configure('Red1.TLabelframe.Label', font=('courier', 12, 'bold'))

        RNNhyperParameterFrame=ttk.Frame(master=hyperParameterFrame)
        RNNhyperParameterFrame.pack(side=LEFT,expand=True,fill="both")
        
        rnn_usecase_frame=ttk.Frame(master=RNNhyperParameterFrame)
        rnn_usecase_frame.pack(side=TOP,expand=False,fill="both")
        
        renn_usecase_var=IntVar()
        
        sapm_not_spam=ttk.Radiobutton(master=rnn_usecase_frame,value=1,variable=renn_usecase_var,text="Spam Not Spam",command=None)
        sapm_not_spam.pack(side=LEFT,expand=False,fill=None)
        
        sentiment=ttk.Radiobutton(master=rnn_usecase_frame,value=2,variable=renn_usecase_var,text="Sentiment Analysis",command=None)
        sentiment.pack(side=LEFT,expand=False,fill=None)
        
        sequence_prediction=ttk.Radiobutton(master=rnn_usecase_frame,value=3,variable=renn_usecase_var,text="Sequence Prediction",command=None)
        sequence_prediction.pack(side=LEFT,expand=False,fill=None) 
        
        lbl=ttk.LabelFrame(master=RNNhyperParameterFrame,text="Hyperparameters",style="Red1.TLabelframe")
        lbl.pack(side=TOP,expand=True,fill="both")

        rnn_lr_lbl=ttk.Label(master=lbl,text="Learning Rate",font=('courier',12,'bold'))
        rnn_lr_lbl.pack(side=LEFT,expand=False,fill=None)
        
        rnn_lr=Entry(master=lbl,bd=5)
        rnn_lr.pack(side=LEFT,expand=False,fill=None)
        
        rnn_dr_lbl=ttk.Label(master=lbl,text="DropoutRate",font=('courier',12,'bold'))
        rnn_dr_lbl.pack(side=LEFT,expand=False,fill=None)
        
        rnn_dr=Entry(master=lbl,bd=5)
        rnn_dr.pack(side=LEFT,expand=False,fill=None)
        
        epo=ttk.Label(master=lbl,text="EPOCH",font=('courier',12,'bold'))
        epo.pack(side=LEFT,expand=False,fill=None)
        
        epo_entry=Entry(master=lbl,bd=5)
        epo_entry.pack(side=LEFT,expand=False,fill=None)
        
        imageFrame.config(text="")
        progressFrame.config(text="")
        
        #return hyperParameterFrame

def is_empty(any_structure):
    if any_structure:
        return False
    else:
        return True

def run_algo():
    global filenamne,imagesLabels,weightLabels,strides,filters,lr,rnn_lr,input_channel
    global l1h,l1w,l1s,l2h,l2w,l2s,l3h,l3w,l3s,l4h,l4w,l4s
    global fc1,fc2,dropout,rb,num_class,var
    global stridelistbox1,stridelistbox2,stridelistbox3,stridelistbox4,epoch_label,steps_label,train_accuracy_label,val_accuracy_label,test_accuracy_label
    
    i=int(var.get())
    if i==1:
        l1hv=int(l1h.get())
        l1wv=int(l1w.get())
        l1sv=int(l1s.get())
        l2hv=int(l2h.get())
        l2wv=int(l2w.get())
        l2sv=int(l2s.get())
        l3hv=int(l3h.get())
        l3wv=int(l3w.get())
        l3sv=int(l3s.get())
        l4hv=int(l4h.get())
        l4wv=int(l4w.get())
        l4sv=int(l4s.get())
        
        num_class_value=int(num_class.get())
        if is_empty(num_class_value):
            num_class_value=5
        
        filters=((l1hv,l1wv,input_channel,l1sv), (l2hv,l2wv,l1sv,l2sv),  (l3hv, l3wv, l2sv, l3sv),  (l4hv, l4wv,l3sv,l4sv))
        
        #print(filters)
        
        all_items=["1,1,1,1","1,2,2,1","1,3,3,1","1,4,4,1"]
        sel_idx1=stridelistbox1.curselection()
        sel_idx2=stridelistbox2.curselection()
        sel_idx3=stridelistbox3.curselection()
        sel_idx4=stridelistbox4.curselection()
        
        if is_empty(sel_idx1):
            sel_idx1=(0,)
        if is_empty(sel_idx2):
            sel_idx2=(0,)
        if is_empty(sel_idx3):
            sel_idx3=(0,)
        if is_empty(sel_idx4):
            sel_idx4=(0,)  
         
        s1=all_items[sel_idx1[0]].split(",")
        s1=[int(i) for i in s1]
        s2=all_items[sel_idx2[0]].split(",")
        s2=[int(i) for i in s2]
        s3=all_items[sel_idx3[0]].split(",")
        s3=[int(i) for i in s3]
        s4=all_items[sel_idx4[0]].split(",")
        s4=[int(i) for i in s4]  
            
        strides=(s1,s2,s3,s4 )
        
        #print(strides)
        lrv=str(lr.get()).strip()
        if is_empty(lrv):
            lrv=0.001
        print("Learning rate being used",lrv)
        drop_rate=str(dropout.get()).strip()
        if is_empty(drop_rate):
            drop_rate=0.001
        
        padding= "SAME" if int(rb.get())==1 else "VALID"
        
        k = threading.Thread(target=kan_train_tf.train,args=(filename, imagesLabels,weightLabels,epoch_label,steps_label,train_accuracy_label,val_accuracy_label,test_accuracy_label,filters,strides,float(lrv),float(drop_rate),padding,(int(fc1.get()),int(fc2.get())),num_class_value,None))
        k.daemon=True
        k.start()
    
    elif i==2:
        global rnn_lr,rnn_dr,epo_entry

        lrv=str(rnn_lr.get()).strip()
        if is_empty(lrv):
            lrv=0.001
        else:
            lrv=float(lrv)

        drv=str(rnn_dr.get()).strip()
        if is_empty(drv):
            drv=0.3
        else:
            drv=float(drv)
        
        epoc=str(epo_entry.get()).strip()        
        if is_empty(epoc):
            epoc=1
        else:
            epoc=int(epoc)
        
        rnn_case=get_rnn_use_case()
        
        
        if rnn_case==1:
            
            k = threading.Thread(target=spam_not_spam_train.train,args=(lrv,epoc,epoch_label,steps_label,train_accuracy_label,val_accuracy_label,test_accuracy_label))
            k.daemon=True
            k.start()
        
        elif rnn_case==2:
            print(filename)
            assert filename != "D:\DL\KannadaData\lessData","Filename is mandatory"
            
            k = threading.Thread(target=sentiment_analysis_movie_batch.train,args=(filename,lrv,drv,epoc,epoch_label,steps_label,train_accuracy_label,val_accuracy_label,test_accuracy_label))
            k.daemon=True
            k.start()
        elif rnn_case==3:
            print(filename)
            assert filename != "D:\DL\KannadaData\lessData","Filename is mandatory"
            
            k = threading.Thread(target=sequence_prediction_lstm_train.train,args=(filename,lrv,drv,epoc,epoch_label,steps_label,train_accuracy_label,val_accuracy_label,test_accuracy_label))
            k.daemon=True
            k.start()

        else:
            pass
        
        

    else:
        pass
        
        

def get_train_page(nb):
        
        s = ttk.Style()
        s.configure('Red.TLabelframe.Label', font=('courier', 15, 'bold'))
        
        subStyle=ttk.Style()
        subStyle.configure('Red.TLabelframe.Label', font=('courier', 10, 'bold'))
        
        buttonStyle=ttk.Style()
        buttonStyle.configure('H.TButton', font=('courier', 12, 'bold')) 
        
        #s.configure('Red.TLabelframe.Label', foreground ='red')
        #s.configure('Red.TLabelframe.Label', background='blue')
    
        global pathLabel,imagesLabels,weightLabels,progressLabel,hyperParameterFrame,train_page,var,epoch_label,steps_label,train_accuracy_label,val_accuracy_label,test_accuracy_label
        global imageFrame,progressFrame
        var=IntVar()
        
        train_page=ttk.Frame(master=nb)
        
        accuracy_frame=ttk.LabelFrame(master=train_page,text="RESULT",style = "Red.TLabelframe")
        accuracy_frame.pack(side=TOP,expand=False,fill="both")
        #label.config()
        
        epoch_label=ttk.Label(master=accuracy_frame, text="",font=("Courier New", 15))
        epoch_label.pack(side=LEFT,expand=False,fill="x")
        
        steps_label=ttk.Label(master=accuracy_frame, text="",font=("Courier New", 15))
        steps_label.pack(side=LEFT,expand=False,fill="x")
        
        train_accuracy_label=ttk.Label(master=accuracy_frame, text="",font=("Courier New", 15))
        train_accuracy_label.pack(side=LEFT,expand=False,fill="x")
        
        val_accuracy_label=ttk.Label(master=accuracy_frame, text="",font=("Courier New", 15))
        val_accuracy_label.pack(side=LEFT,expand=False,fill="x")
        
        test_accuracy_label=ttk.Label(master=accuracy_frame, text="",font=("Courier New", 15))
        test_accuracy_label.pack(side=LEFT,expand=False,fill="x")
        
        #algoFrame=LabelFrame(master=train_page, bg="green",text="ALGORITHMS")
        algoFrame=ttk.LabelFrame(master=train_page, text="ALGORITHMS",style = "Red.TLabelframe")
        algoFrame.pack(side=TOP,expand=False,fill="both")
        
        rbs=[ttk.Radiobutton(master=algoFrame,value=v,variable=var,text=text,command=radio_button_cmd) for text,v in ALGOS]

        for rb in rbs:
            rb.pack(side=LEFT,anchor=N,expand=True,fill="both")
      
        #controlFrame=ttk.LabelFrame(master=train_page, bg="grey",text="SELECTION")
        controlFrame=ttk.LabelFrame(master=train_page, text="SELECTION",style = "Red.TLabelframe")
        controlFrame.pack(side=TOP,expand=False,fill="both")
        
        browseButton=ttk.Button(master=controlFrame,text="Select Data Folder",command=browse_button_press,style="H.TButton")
        browseButton.pack(side=LEFT,anchor=N,expand=True,fill="both")
        
        pathLabel=ttk.Label(master=controlFrame,text="",font=("Calibri", 20))
        pathLabel.pack(side=LEFT,anchor=N,fill="both",expand=True)
        
        runButton=ttk.Button(master=controlFrame,text="Run",command=run_algo,style="H.TButton")
        runButton.pack(side=LEFT,anchor=N,expand=True,fill="both")
        
        hyperParameterFrame=get_hyperParameterFrame()  
    
        """
        pFrame1=ttk.Frame(master=train_page)
        pFrame1.pack(side=TOP,expand=True,fill="both")
    
        pFrame2=ttk.Frame(master=train_page)
        pFrame2.pack(side=TOP,expand=True,fill="both")
        """
        
        imageFrame=ttk.LabelFrame(master=train_page,text="ACTIVATION MAP Visualization",style = "Red.TLabelframe")
        imageFrame.pack(side=TOP,expand=True,fill="both")
        
        #ims=[  ImageTk.PhotoImage(image=Image.fromarray(np.random.randn(50,50,3),"RGB"),size=(50,50,3)) for i in range(50) ]
        activationframes=[ttk.Frame(master=imageFrame) for i in range(10)]
        
        for i in activationframes:
            i.pack(side=TOP,expand=True,fill="both")
            
        for ix in range(50):        
            imagesLabels.append(Label(master=activationframes[ix%5],image=None))
            imagesLabels[-1].pack(side=LEFT,expand=True,fill="both")
        
        pathLabel=ttk.Label(master=controlFrame,text="")
        pathLabel.pack(side=LEFT,anchor=N,fill="both",expand=True)
        
        
        progressFrame=ttk.LabelFrame(master=train_page,text="WEIGHT Visualization",style = "Red.TLabelframe")
        progressFrame.pack(side=TOP,expand=True,fill="both")    
        
        weightframes=[ttk.Frame(master=progressFrame) for i in range(4)]
        
        for i in weightframes:
            i.pack(side=TOP,expand=True,fill="both")
          

        for ix in range(40):
            weightLabels.append(Label(master=weightframes[ix%4],image=None))
            weightLabels[-1].pack(side=LEFT,expand=True,fill="both")
        
        return train_page

def modselbutton_press():
    global selected_model_name,model_name_label
    selected_model_name=filedialog.askopenfilename(initialdir = "D:\\DL\\KannadaData\\lessData",title = "Select file",filetypes = (("model_config","*.model_config"),("all files","*.*")))
    model_name_label.config(text=selected_model_name)

def valimselbutton_press():
    global val_im,valim_name_label
    val_im=filedialog.askopenfilename(initialdir = "D:\\DL\\KannadaData\\lessData",title = "Select file",filetypes = (("png files","*.png"),("all files","*.*")))
    valim_name_label.config(text=val_im)

def inf_button_press():
    global val_result_label,val_drop,selected_model_name,val_im
        
    k = threading.Thread(target=do_inference.inference,args=(val_result_label,selected_model_name,val_im,0.001,1))
    k.start()    
    
    #do_inference.inference(val_result_label,selected_model_name,val_im,0.001,val_droprate)


def get_validation_page(nb):
    
    global model_name_label,valim_name_label,val_result_label,val_drop,val_image__selection_frame,valimselbutton,val_frame,val_container_frame
    
    b=ttk.Style()
    b.configure('Gen.TLabelframe.Label', font=('courier', 15, 'bold'))
    
    buttonStyle=ttk.Style()
    buttonStyle.configure('Generic.TButton', font=('courier', 12, 'bold'))
  
    val_container_frame = ttk.Frame(master=nb)
    val_frame=ttk.Frame(master=val_container_frame)
    val_frame.pack(side=TOP,expand=True,fill="both")
    
    model_selection_frame=ttk.LabelFrame(master=val_frame,text="Select Model",style="Gen.TLabelframe")
    model_selection_frame.pack(side=TOP,expand=False,fill="both")
    modselbutton=ttk.Button(master=model_selection_frame,text="Select model file",command=modselbutton_press,style="Generic.TButton")
    modselbutton.pack(side=LEFT,anchor=N,expand=False,fill="y")
    model_name_label=ttk.Label(master=model_selection_frame,text="")
    model_name_label.pack(side=LEFT,anchor="center",fill="x",expand=True)
    
    
    val_image__selection_frame=ttk.LabelFrame(master=val_frame,text="Select Image",style="Gen.TLabelframe")
    val_image__selection_frame.pack(side=TOP,expand=False,fill="both")
    valimselbutton=ttk.Button(master=val_image__selection_frame,text="Select Image file",command=valimselbutton_press,style="Generic.TButton")
    valimselbutton.pack(side=LEFT,anchor=N,expand=False,fill="y")
    valim_name_label=ttk.Label(master=val_image__selection_frame,text="")
    valim_name_label.pack(side=LEFT,anchor="center",fill="x",expand=True)
    
    param_frame=ttk.LabelFrame(master=val_frame,text="Parameters",style="Gen.TLabelframe")
    param_frame.pack(side=TOP,expand=False,fill="both")
    
    #val_lr_label=ttk.Label(master=param_frame,text="Learning Rate: ",font=('courier',12,'bold'))
    #val_lr_label.pack(side=LEFT,expand=True,fill="both")
    
    #val_lr=Entry(master=param_frame,bd=5,font=('courier',12,'bold'))
    #val_lr.pack(side=LEFT,expand=True,fill="both")
    
    #val_drop_label=ttk.Label(master=param_frame,text="Dropout Rate: ",font=('courier',12,'bold'))
    #val_drop_label.pack(side=LEFT,expand=True,fill="both")
    
    #val_drop=Entry(master=param_frame,bd=5,font=('courier',12,'bold'))
    #val_drop.pack(side=LEFT,expand=True,fill="both")
    
    inf_frame=ttk.Frame(master=val_frame)
    inf_frame.pack(side=TOP,expand=False,fill="both")
    inf_button=ttk.Button(master=inf_frame,text="Run Inference",command=inf_button_press,style="Generic.TButton")
    inf_button.pack(side=LEFT,anchor="center",expand=True,fill="y")
        
    val_result_frame=ttk.LabelFrame(master=val_frame,text="Result",style="Gen.TLabelframe")
    val_result_frame.pack(side=TOP,expand=True,fill="both")
    val_result_label=ttk.Label(master=val_result_frame,text="",font=("Calibri", 20))
    val_result_label.pack(side=LEFT,anchor="center",fill="x",expand=True)
    
    
    
    return val_container_frame

def main():
    root=Tk()
    root.title("Deep Learning Toolbox")
    
    global train_page,stridelistbox1
    nb = ttk.Notebook(root)
    
    train_page=get_train_page(nb)
    
    validation_page=get_validation_page(nb)
    #test_page=ttk.Frame(master=nb)
    
    nb.add(train_page,text="Train/Validation/Test")
    nb.add(validation_page,text="Inference")
    #nb.add(test_page,text="Test")
    
    nb.pack(expand=1, fill="both")
    
    root.minsize(1000,700)
        
    root.mainloop()

if __name__=="__main__":
    main()