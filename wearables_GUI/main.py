# Required Modules
from Tkinter import *
from tkFileDialog import *
from model_fn import *
import os

# Prompt for selecting file
def load_file(path,name):
    fname = askopenfilename(filetypes=[("CSV files","*.csv")])
    path.set(fname)
    name.set(os.path.split(fname)[1])
    return

def set_model_flag(model,flag):
    if model=='ANN':
        flag.set(0)
    elif model=='SVM':
        flag.set(1)
    elif model=='Decision Tree':
        flag.set(2)
    return

# Create GUI
gui = Tk()
gui.geometry("700x700")
gui.wm_title("Wearables GUI")

# dimensions
pad = 0.005
sub_title = 'Times 20 bold'
sub_text = 'Times 12'
sub_option = 'Times 10'

# PART 1 : Title and Description
part1_frame = Frame(gui,highlightbackground="black",highlightthickness=2)
part1_frame.place(relx=pad,rely=pad,relheight=0.3-1.5*pad,relwidth=1-2*pad)
Label(part1_frame,text=part1_title,font='Times 25 bold').pack()
Label(part1_frame,text=part1_description,font='Times 14').pack()
Button(part1_frame,text="Instructions",font=sub_option,command=lambda:displ_instructions(),width=14).pack()

# part 2 : Choosing model type
part2_frame = Frame(gui,highlightbackground="black",highlightthickness=2)
part2_frame.place(relx=pad,rely=0.3+0.5*pad,relheight=0.35-pad,relwidth=0.5-1.5*pad)
Label(part2_frame,text=part2_title,font=sub_title).pack()
Label(part2_frame,text=part2_description,font=sub_text).pack()
train_flag = IntVar()
Radiobutton(part2_frame,text="Pretrained",font=sub_option,variable=train_flag, value=0).pack()
Radiobutton(part2_frame,text="Train Model",font=sub_option,variable=train_flag, value=1).pack()

# part 3 : Select type of classifier
part3_frame = Frame(gui,highlightbackground="black",highlightthickness=2)
part3_frame.place(relx=0.5+0.5*pad,rely=0.3+0.5*pad,relheight=0.35-pad,relwidth=0.5-1.5*pad)
Label(part3_frame,text=part3_title,font=sub_title).pack()
Label(part3_frame,text=part3_description,font=sub_text).pack()
model_flag = IntVar()
model_val = StringVar()
model_val.set('ANN')
model_flag.set(0)
optionList = ('ANN','SVM','Decision Tree')
OptionMenu(part3_frame,model_val,*optionList,command=lambda x:set_model_flag(model_val.get(),model_flag)).pack()

# part 4 ; enter data
part4_frame = Frame(gui,highlightbackground="black",highlightthickness=2)
part4_frame.place(relx=pad,rely=0.65+0.5*pad,relheight=0.35-1.5*pad,relwidth=0.5-1.5*pad)
Label(part4_frame,text=part4_title,font=sub_title).pack()
Label(part4_frame,text=part4_description,font=sub_text).pack()

#SET TRAIN FLAG, HIDE THE BUTTON IF PRETRAINED IS SELECTED
train_filepath = StringVar()
train_filename = StringVar()
Button(part4_frame,text="Open Train File",font=sub_option,command=lambda:load_file(train_filepath,train_filename),width=10).pack()
Label(part4_frame, textvariable =train_filename).pack()
test_filepath = StringVar()
test_filename = StringVar()
Button(part4_frame,text="Open Validation File",font=sub_option,command=lambda:load_file(test_filepath,test_filename),width=14).pack()
Label(part4_frame, textvariable =test_filename).pack()

#part 5 : Training and Classification
part5_frame = Frame(gui,highlightbackground="black",highlightthickness=2)
part5_frame.place(relx=0.5+0.5*pad,rely=0.65+0.5*pad,relheight=0.35-1.5*pad,relwidth=0.5-1.5*pad)
Label(part5_frame,text=part5_title,font=sub_title).pack()
Label(part5_frame,text=part5_description,font=sub_text).pack()
status = StringVar()
Button(part5_frame,text="Classify !",font=sub_option,command=lambda:get_output(train_flag.get(),model_flag.get(),train_filepath.get(),test_filepath.get(),status),width=10).pack()
Label(part5_frame, textvariable =status).pack()

gui.mainloop()
