from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn import tree
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import metrics


df = pd.read_csv('btltest.csv')
X = np.array(df[['age','height','won','lost','drawn','KOs']].values)    
y = np.array(df['result'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , shuffle = True)
#id3
id3 = DecisionTreeClassifier(criterion='entropy',max_depth=4, random_state=42)
id3.fit(X_train, y_train)
#svm
svm = SVC()
svm.fit(X_train, y_train) 
#Neuron
neuron = MLPClassifier(random_state=1, activation='logistic', max_iter=1000).fit(X_train, y_train)
#print(clf.predict(X_Test))
neuron.score(X_train, y_train)
#form
form = Tk()
form.title("Dự đoán chiến thắng của Boxer:")
form.geometry("1000x500")



lable_ten = Label(form, text = "Nhập thông tin Boxer:", font=("Arial Bold", 10), fg="red")
lable_ten.grid(row = 1, column = 1, padx = 40, pady = 10)

lable_buying = Label(form, text = "Tuổi:")
lable_buying.grid(row = 2, column = 1, padx = 40, pady = 10)
textbox_buying = Entry(form)
textbox_buying.grid(row = 2, column = 2)

lable_maint = Label(form, text = "Chiều cao:")
lable_maint.grid(row = 3, column = 1, pady = 10)
textbox_maint = Entry(form)
textbox_maint.grid(row = 3, column = 2)

lable_doors = Label(form, text = "Số trận thắng:")
lable_doors.grid(row = 4, column = 1,pady = 10)
textbox_doors = Entry(form)
textbox_doors.grid(row = 4, column = 2)

lable_persons = Label(form, text = "Số trận thua:")
lable_persons.grid(row = 5, column = 1, pady = 10)
textbox_persons = Entry(form)
textbox_persons.grid(row = 5, column = 2)

lable_lug_boot = Label(form, text = "Số lần vô địch:")
lable_lug_boot.grid(row = 6, column = 1, pady = 10 )
textbox_lug_boot = Entry(form)
textbox_lug_boot.grid(row = 6, column = 2)

lable_safety = Label(form, text = "Số lần hạ KO:")
lable_safety.grid(row = 7, column = 1, pady = 10 )
textbox_safety = Entry(form)
textbox_safety.grid(row = 7, column = 2)



#neuron
#neuron
#dudoanneurontheotest
y_neuron = neuron.predict(X_test)
lbl1 = Label(form)
lbl1.grid(column=1, row=8)
lbl1.configure(text="Tỉ lệ dự đoán đúng của Neural Network: "+'\n'
                           +"Precision: "+str(precision_score(y_test, y_neuron, average='macro')*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test, y_neuron, average='macro')*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test, y_neuron, average='macro')*100)+"%"+'\n')
def dudoanneuron():
    buying = textbox_buying.get()
    maint = textbox_maint.get()
    doors = textbox_doors.get()
    persons = textbox_persons.get()
    lug_boot =textbox_lug_boot.get()
    safety =textbox_safety.get()
    if((buying == '') or (maint == '') or (doors == '') or (persons == '') or (lug_boot == '')or (safety == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([buying,maint,doors,persons,lug_boot,safety],dtype=np.float64).reshape(1, -1)
        y_kqua = neuron.predict(X_dudoan)
        lbl.configure(text= y_kqua)
button_neuron = Button(form, text = 'Kết quả dự đoán theo Neural', command = dudoanneuron)
button_neuron.grid(row = 9, column = 1, pady = 20)
lbl = Label(form, text="...")
lbl.grid(column=2, row=9)

def khanangneuron():
    y_neuron = neuron.predict(X_test)
    dem=0
    for i in range (len(y_neuron)):
        if(y_neuron[i] == y_test[i]):
            dem= dem+1
    count = (dem/len(y_neuron))*100
    lbl1.configure(text= count)
button_neuron1 = Button(form, text = 'Khả năng dự đoán đúng ', command = khanangneuron)
button_neuron1.grid(row = 10, column = 1, padx = 30)
lbl1 = Label(form, text="...")
lbl1.grid(column=2, row=10)
#svm
y_svm=svm.predict(X_test)
lblsvm = Label(form)
lblsvm.grid(column=5, row=8)
lblsvm.configure(text="Tỉ lệ dự đoán đúng của SVM: "+'\n'
                           +"Precision: "+str(precision_score(y_test, y_svm, average='macro')*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test, y_svm, average='macro')*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test, y_svm, average='macro')*100)+"%"+'\n')

def dudoansvm():
    buying = textbox_buying.get()
    maint = textbox_maint.get()
    doors = textbox_doors.get()
    persons = textbox_persons.get()
    lug_boot =textbox_lug_boot.get()
    safety =textbox_safety.get()
    if((buying == '') or (maint == '') or (doors == '') or (persons == '') or (lug_boot == '')or (safety == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:     
        X_dudoan = np.array([buying,maint,doors,persons,lug_boot,safety],dtype=np.float64).reshape(1, -1)
        y_kqua = neuron.predict(X_dudoan)
        lblsvm.configure(text= y_kqua)     
button_svm = Button(form, text = 'Kết quả dự đoán theo SVM', command = dudoansvm)
button_svm.grid(row = 9, column = 5, pady = 20)
lblsvm = Label(form, text="...")
lblsvm.grid(column=6, row=9)
def khanangsvm():
    y_svm = svm.predict(X_test)
    dem=0
    for i in range (len(y_id3)):
        if(y_svm[i] == y_test[i]):
            dem= dem+1
    count = (dem/len(y_id3))*100
    lblsvm1.configure(text= count)
button_svm = Button(form, text = 'Khả năng dự đoán đúng ', command = khanangsvm)
button_svm.grid(row = 10, column = 5, padx = 30)
lblsvm1 = Label(form, text="...")
lblsvm1.grid(column=6, row=10)                
#id3
#dudoanid3test
y_id3 = id3.predict(X_test)
lbl3 = Label(form)
lbl3.grid(column=3, row=8)
lbl3.configure(text="Tỉ lệ dự đoán đúng của ID3: "+'\n'
                           +"Precision: "+str(precision_score(y_test, y_id3, average='macro')*100)+"%"+'\n'
                           +"Recall: "+str(recall_score(y_test, y_id3, average='macro')*100)+"%"+'\n'
                           +"F1-score: "+str(f1_score(y_test, y_id3, average='macro')*100)+"%"+'\n')
def dudoanid3():
    buying = textbox_buying.get()
    maint = textbox_maint.get()
    doors = textbox_doors.get()
    persons = textbox_persons.get()
    lug_boot =textbox_lug_boot.get()
    safety =textbox_safety.get()
    if((buying == '') or (maint == '') or (doors == '') or (persons == '') or (lug_boot == '')or (safety == '')):
        messagebox.showinfo("Thông báo", "Bạn cần nhập đầy đủ thông tin!")
    else:
        X_dudoan = np.array([buying,maint,doors,persons,lug_boot,safety]).reshape(1, -1)
        y_kqua = id3.predict(X_dudoan)
        lbl2.configure(text= y_kqua)
    
button_id3 = Button(form, text = 'Kết quả dự đoán theo ID3', command = dudoanid3)
button_id3.grid(row = 9, column = 3, pady = 20)
lbl2 = Label(form, text="...")
lbl2.grid(column=4, row=9)

def khanangid3():
    y_id3 = id3.predict(X_test)
    dem=0
    for i in range (len(y_id3)):
        if(y_id3[i] == y_test[i]):
            dem= dem+1
    count = (dem/len(y_id3))*100
    lbl3.configure(text= count)
button_id31 = Button(form, text = 'Khả năng dự đoán đúng ', command = khanangid3)
button_id31.grid(row = 10, column = 3, padx = 30)
lbl3 = Label(form, text="...")
lbl3.grid(column=4, row=10)


form.mainloop()
