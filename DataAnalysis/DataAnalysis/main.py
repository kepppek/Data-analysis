# -*- coding: utf-8 -*-
from pickle import FALSE
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
import sys

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer,QSize
#from PyQt5.Qt import QStandardItemModel, QStandardItem

import numpy as np
import openpyxl
import time
import os

# метрики
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,mean_absolute_error,classification_report
from sklearn import metrics

# алгоритмы и средство визуализации
from classification.k_nearest_neighbors import *
from classification.GaussianProcesses import*
from classification.mySGDClassifier import *
from classification.SVM import *
from classification.D_Tree import *
from classification.N_Byes import *
from classification.Random_Forest import *
from clustering.agglomerative_clustering import *
from clustering.Affinity_Propagation import *
from clustering.K_Means import *
from clustering.OPTICS import *
from clustering.Mean_Shift import *
from clustering.Spectral import *
from clustering.DBScan import *
from clustering.myBirch import *
from chart.tools import *

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# для окон ошибок
from PyQt5.QtWidgets import QMessageBox

# для парсинга пути к файлу
from pathlib import Path

# dataframe
import pandas as pd

# надо для графиков
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

Form, _ = uic.loadUiType("form.ui")

# для изменения параметров текста справки
class StandardItem(QStandardItem):
    def __init__(self,txt='',font_size=12,set_bold=False,color=QColor(0,0,0)):
        super().__init__()
        fnt = QFont("Open Sans",font_size)
        fnt.setBold(set_bold)

        self.setEditable(False)
        self.setForeground(color)
        self.setFont(fnt)
        self.setText(txt)
        self.setSizeHint(QSize(0,(int)((len(txt)/(int(QApplication.primaryScreen().size().width()/2)/1.4/font_size)+1)*(font_size*2))))

class Calculate(QThread):
    mySignal = pyqtSignal(str)                       
    
    def __init__(self) -> None:
        super().__init__()
        self.table = None  # X
        self.algorithm = None   # название алгоритма
        self.target = None   # Y
        self.result = None # результат
        self.parametrs = None    # параметры алгоритма 

    def run(self):
        try:
            # Петров
            if self.algorithm == "Nearest_Neighbors":
                 self.result = Nearest_Neighbors(self.table,self.target,self.parametrs)

            # Кочетков/Петров
            elif self.algorithm == "Agglomerative":
                 self.result = Agglomerative_Clustering(self.table,self.parametrs)

            # Великов - кластеризация     
            elif self.algorithm == "Affinity_Propagation":
                 self.result = Affinity_Propagation(self.table, self.parametrs)

            # Великов/Кочетков - классификация
            elif self.algorithm == "SVM":
                 self.result = mySVM(self.table,self.target,self.parametrs)

            # Чугунов/Жендосян - кластеризация
            elif self.algorithm == "OPTICS":
                 self.result = Optics(self.table, self.parametrs)

            # Чугунов/Гуляев - классификация
            elif self.algorithm == "SGDClassifier":
                 self.result = mySGDClassifier(self.table,self.target,self.parametrs)

            # Гуляев - кластеризация
            elif self.algorithm == "K_Means":
                 self.result = K_Means(self.table, self.parametrs)

            # Жендосян - классификация
            elif self.algorithm == "Gaussian_Process":
                 self.result = gaussianProcessClassifier(self.table,self.target,self.parametrs)

            
            # Базовые методы (Великов, Гуляев, Минаков, Чугунов / 01.07.2022)     
            elif self.algorithm == "Mean_Shift":
                 self.result = Mean_Shift(self.table, self.parametrs)

            elif self.algorithm == "Spectral":
                 self.result = Spectral(self.table, self.parametrs)

            elif self.algorithm == "myBirch":
                 self.result = myBirch(self.table, self.parametrs)

            elif self.algorithm == "DBScan":
                 self.result = DBScan(self.table, self.parametrs)
            
            elif self.algorithm == "D_Tree":
                 self.result = D_Tree(self.table,self.target,self.parametrs)

            elif self.algorithm == "N_Byes":
                 self.result = N_Byes(self.table,self.target,self.parametrs)

            elif self.algorithm == "Random_Forest":
                 self.result = Random_Forest(self.table,self.target,self.parametrs)

            self.mySignal.emit('Success')
            #Canvas.timer.stop()
        except Exception as err:
            self.mySignal.emit(str(err))                          


class Canvas(QMainWindow):
    def __init__(self):
        super(Canvas, self).__init__()
        self.ui = Form()
        self.ui.setupUi(self)        
        self.flag = False # был ли изменен входной dataset
        self.path = ""    # путь к файлу
        self.thread = Calculate() # поток в котором будут происходить вычисления
        self.thread.mySignal.connect(self.Result)  # подключаем поток к методу Result
        self.figureChart = None   # график графика
        self.canvasChart = None   
        self.figureMetrics = None  # график метрик
        self.canvasMetrics = None   
        self.dataset = None # исходная таблица
        self.ConfigureInterface() # запуск первичной настройки различных параметров



        # записываем значение в счетчик миллисекунд
        self.ui.timeViewer.setText("0")
        self.ui.timeViewer.setAlignment(Qt.AlignCenter)

        # скрываем вкладку метрик
        #self.ui.tabWidget.setTabVisible(2,False)



        # Таймер
        self.timer = QTimer(self)
        self.time = 0


    def SetHelp(self):
        treeModel = QStandardItemModel()
        rootNode = treeModel.invisibleRootItem()
        self.ui.treeView.setHeaderHidden(True)
        self.ui.treeView.setWordWrap(True)
       # self.ui.treeView.setUniformRowHeights(False)
       # print(self.ui.treeView.height())
        t1 = StandardItem('Руководство пользователя',16)
        rootNode.appendRow(t1)

        t11 = StandardItem('''Для запуска алгоритмов потребуется:
1) Выбрать файл формата .csv или .xlsx.В файле должен присутствовать столбец с названием "Истина" - он автоматически будет определяться как кластер
2) Выбрать тип моделирования: Классификация или Кластеризация
3) Выбрать алгоритм для вычислений
4) Ввести параметры для алгоритма
5) Нажать кнопку запуска
P.S. На вкладке "График", есть специальная панель для замены цветов кластеров местами.Это необходимо т.к. алгоритмы выдают названия кластерам в случайном порядке P.S.2. Большинство ошибок обрабатывается.Так если считывание файла происходит с ошибкой, значит файл содержит некорректные данные(текстовые и т.д.)''',12)    
        t1.appendRow(t11);

        t2 = StandardItem('Метрики',16)
        rootNode.appendRow(t2)
        ###############
        t21 = StandardItem('Accuracy',12)
        t2.appendRow(t21)

        t211 = StandardItem('доля правильных ответов алгоритма',12)
        t21.appendRow(t211)
        ###############
        t22 = StandardItem('Precision',12)
        t2.appendRow(t22)

        t221 = StandardItem('можно интерпретировать как долю объектов, названных классификатором положительными и при этом действительно являющимися положительными',12)
        t22.appendRow(t221)
        ###############
        t23 = StandardItem('Recall',12)
        t2.appendRow(t23)

        t231 = StandardItem('показывает, какую долю объектов положительного класса из всех объектов положительного класса нашел алгоритм',12)
        t23.appendRow(t231)
        ###############
        t24 = StandardItem('Error',12)
        t2.appendRow(t24)

        t241 = StandardItem('доля неправильных ответов алгоритма',12)
        t24.appendRow(t241)
        ###############
        t25 = StandardItem('F1',12)
        t2.appendRow(t25)

        t251 = StandardItem('фактически это просто среднее гармоническое величин P и R',12)
        t25.appendRow(t251)
        ###############

        self.ui.treeView.setModel(treeModel) 


  
        self.ui.tab_3.layout = QVBoxLayout() 
        self.ui.tab_3.layout.addWidget(self.ui.treeView)
        self.ui.tab_3.setLayout(self.ui.tab_3.layout)
      #  print(self.ui.tab_3.layout.frameGeometry().width())


    def ClearAllFillingData(self):
        self.SwitchButtonsEnabled(False)
        self.ClearAlgorithm()
        self.ClearResult()         
        self.ui.cbModel.clear()
        self.ui.cbAlgorithm.clear()
        self.ui.cbData.clear()
        self.ui.cbTarget.clear()
        self.ui.tbPath.setText("")
        self.ui.tbNameFile.setText("")


    def ClearResult(self):
        self.figureChart.clear()
        self.canvasChart.draw()
        self.ui.cbX.clear()
        self.ui.cbY.clear()
        self.ui.cbZ.clear()
        self.ui.cbColor.clear()
        self.ui.tbMetrics.setText("")
        self.ui.tbStatus.setText("Ожидаю...")
        self.ui.cbCluster1.clear()
        self.ui.cbCluster2.clear()   
        self.figureMetrics.clear()
        self.canvasMetrics.draw()
        #self.ui.cbYpred.clear()
        #self.ui.cbYtrue.clear()
        self.thread.result = None
        self.thread.target = None
        self.thread.table = None
        self.flag = False
        self.ui.buttonCopyToExcel.setEnabled(False)
        self.ui.buttonSwap.setEnabled(False)
        #self.ui.tabWidget.setTabVisible(2,False)
        #self.FillMetrics(False)

    def SwitchButtonsEnabled(self,flag):
        self.ui.buttonCopyToExcel.setEnabled(flag)
        self.ui.buttonLaunch.setEnabled(flag)
        self.ui.buttonSwap.setEnabled(flag)
        self.ui.buttonDraw.setEnabled(flag)
        self.ui.buttonMetrics.setEnabled(flag)


    def OpenFile(self):     
        self.path = QFileDialog.getOpenFileName(self,  'Выберите файл',directory = "./data/",filter = "Excel (*.csv *.xlsx)")[0] 
        if self.path=="":  # если отменили открытие файла
            return
        try:
          self.ClearAllFillingData()
          p = Path(self.path)

          if p.suffix == ".csv":    # "ISO-8859-1"
              self.dataset = pd.read_csv(self.path,sep=";",decimal = ".",encoding = "utf-8")  
          elif p.suffix == ".xlsx":
              self.dataset = pd.read_excel(self.path)  
          else:
             raise Exception("Неверное расширение!")    

          # меняем запятую на точку (если есть) иначе будут проблемы
          self.dataset = self.dataset.astype(str).apply(lambda x: x.str.replace(',','.', regex=True))
          # преобразуем из str обратно в float
          self.dataset  = self.dataset.apply(pd.to_numeric, errors='ignore')# coerce

          # если в dataset есть столбцы с именем Y_Pred, то переименуем их
          #self.dataset.rename(columns={'Y_Pred': 'Y_Pred.0'}, inplace=True)



          #if "Прогноз" in self.dataset.columns:
          #     self.dataset = self.dataset.drop("Прогноз",axis = 1)

          #if not "Истина" in self.dataset.columns:
          #     raise Exception("Отсутствует столбец \"Истина\"!")    

          self.ui.tbNameFile.setText(p.name)
          self.ui.tbPath.setText(self.path)
          self.FillModel()
          self.FillDataSet()
          self.FillTarget()
          self.ui.buttonDraw.setEnabled(True)
          self.ui.buttonMetrics.setEnabled(True)
        except Exception as err:
          self.ClearAllFillingData()
          QMessageBox.about(self, "Ошибка!", "Не удалось открыть файл!\n"+str(err))


    def FillDataSet(self):
        self.ui.cbData.addItem(' ')
        i = 1
        for name in self.dataset.columns:
            if name != "Истина":
             self.ui.cbData.addItem(name)
             item = self.ui.cbData.model().item(i, self.ui.cbData.modelColumn())
             item.setCheckState(Qt.Checked) 
             i=i+1


    def SwitchTarget(self):
        self.flag = False
            

    def SwitchModel(self):
        #self.ui.cbAlgorithm.clear()  
        #self.ui.cbAlgorithm.addItem("")
        if self.ui.cbModel.currentText() == "Классификация":
            if not "Истина" in self.dataset.columns:
               self.ui.cbModel.setCurrentIndex(0)
               QMessageBox.about(self, "Ошибка!", "Отсутствует столбец \"Истина\"!")
               return 
            self.ui.cbAlgorithm.clear()  
            self.ui.cbTarget.show()
            self.ui.label_9.show()
            self.ui.cbAlgorithm.addItem("Nearest_Neighbors")
            self.ui.cbAlgorithm.addItem('SGDClassifier')
            self.ui.cbAlgorithm.addItem('SVM')
            self.ui.cbAlgorithm.addItem('Gaussian_Process')
            self.ui.cbAlgorithm.addItem('D_Tree')
            self.ui.cbAlgorithm.addItem('N_Byes')
            self.ui.cbAlgorithm.addItem('Random_Forest')

            ###
        elif self.ui.cbModel.currentText() == "Кластеризация":
            self.ui.cbAlgorithm.clear()  
            self.ui.cbTarget.hide()
            self.ui.label_9.hide()
            self.ui.cbAlgorithm.addItem("Agglomerative")
            self.ui.cbAlgorithm.addItem('Affinity_Propagation')
            self.ui.cbAlgorithm.addItem('OPTICS')
            self.ui.cbAlgorithm.addItem('K_Means')
            self.ui.cbAlgorithm.addItem('Mean_Shift')
            self.ui.cbAlgorithm.addItem('Spectral')
            self.ui.cbAlgorithm.addItem('myBirch')
            self.ui.cbAlgorithm.addItem('DBScan')
            ###        
        self.SwitchAlgorithm()


    def FillTarget(self):
          self.ui.cbTarget.addItem("")
          #for name in self.dataset.columns:
          #  self.ui.cbTarget.addItem(name)
          self.ui.cbTarget.addItem("Истина")
          self.ui.cbTarget.setCurrentIndex(1)



    def FillXYZ(self):
        self.ui.cbX.clear()
        self.ui.cbY.clear()
        self.ui.cbZ.clear()
        self.ui.cbColor.clear()
        self.ui.cbX.addItem("")
        self.ui.cbY.addItem("")
        self.ui.cbZ.addItem("")
        self.ui.cbColor.addItem("")
        if self.thread.result is not None:
         for name in self.thread.result.columns:
           self.ui.cbX.addItem("Р: "+name)
           self.ui.cbY.addItem("Р: "+name)
           self.ui.cbZ.addItem("Р: "+name)
           self.ui.cbColor.addItem("Р: "+name)
        if self.dataset is not None:
         for name in self.dataset.columns:
           self.ui.cbX.addItem("И: "+name)
           self.ui.cbY.addItem("И: "+name)
           self.ui.cbZ.addItem("И: "+name)
           self.ui.cbColor.addItem("И: "+name)


    def FillMetrics(self,flag):
        print("123")
       # if flag and self.trueColumn:
       #   self.ui.tabWidget.setTabVisible(2,True)
       # else:
       #   self.ui.tabWidget.setTabVisible(2,False)
       # self.ui.tabWidget.setTabVisible(2,True)
       #a = 5
       # if self.thread.result is not None:
       #  pred = False
       #  true = False
       #  for name in self.thread.result.columns:
       #      if(name=="Истина"):
       #          true = True
       #      if(name=="Прогноз"):
       #          pred = True
       #  if pred==True and true == True:
       #    self.CalculateMetrics(self)
       #    self.ui.tabWidget.setTabVisible(2,True)
       #  else:
       #    self.ui.tabWidget.setTabVisible(2,False) 
       # else:
       #  self.ui.tabWidget.setTabVisible(2,False) 

        #self.ui.cbYtrue.clear()
        #self.ui.cbYpred.clear()
        #self.ui.cbYtrue.addItem("")
        #self.ui.cbYpred.addItem("")
        #if self.thread.result is not None:
        # for name in self.thread.result.columns:
        #   self.ui.cbYtrue.addItem("П: "+name)
        #   self.ui.cbYpred.addItem("П: "+name)
        #if self.dataset is not None:
        # for name in self.dataset.columns:
        #   self.ui.cbYtrue.addItem("И: "+name)
        #   self.ui.cbYpred.addItem("И: "+name)
        #   #☐ ☑


    def FillModel(self):
        self.ui.cbModel.addItem('Кластеризация')
        self.ui.cbModel.addItem('Классификация')    
        self.SwitchModel()


    def ClearAlgorithm(self):
        for i in reversed(range(self.ui.tab.layout.count())): 
            widgetToRemove = self.ui.tab.layout.itemAt(i).widget()
            self.ui.tab.layout.removeWidget(widgetToRemove) ##??? иначе может ошибка возникнуть
            widgetToRemove.setParent(self.ui.container)


    def SwitchAlgorithm(self):
        if self.ui.cbAlgorithm.currentText()=="":
            self.ui.buttonLaunch.setEnabled(False)
        else:
            self.ui.buttonLaunch.setEnabled(True)
            self.thread.algorithm = self.ui.cbAlgorithm.currentText()
        # очищаем
        self.ClearAlgorithm()

        # заполняем
        if self.ui.cbAlgorithm.currentText() != "":
            for container in self.ui.container.children():            
                if self.ui.cbAlgorithm.currentText() == container.objectName():
                     self.ui.tab.layout.addWidget(container)
                     self.ui.tab.setLayout(self.ui.tab.layout)
                     break

        self.ClearResult() # очищаем ВСЕ
        self.FillXYZ()   # заполняем исходными данными
       # self.FillMetrics(False)  # заполняем исходными данными

 
    # вызывается после завершения вычислений
    def Result(self, status):
            self.SwitchEnabledAll(True)
            self.timer.stop()
            try:
               if status == 'Success':
                self.ui.tbStatus.setText("Успех!") 
                self.ui.buttonSwap.setEnabled(True)
                self.ui.buttonCopyToExcel.setEnabled(True)
                self.FillSwap()
                self.FillXYZ()
                #self.FillMetrics(True)
               else:
                  raise Exception(status)             
            except Exception as err:
              self.ui.buttonSwap.setEnabled(False)
              self.ui.buttonCopyToExcel.setEnabled(False)
              self.ClearResult()
              self.FillXYZ()
              #self.FillMetrics(False)
              self.ui.tbStatus.setText("Ошибка вычислений:\n"+str(err)) 



    def CalculateMetrics(self):
         try:
             #if self.ui.cbYtrue.currentText() == "":
             # raise Exception("Кластер 1 не выбран!") 
             #
             #if self.ui.cbYpred.currentText() == "":
             # raise Exception("Кластер 2 не выбран!") 

            # Ytrue =  self.CheckField("Истина")
            # Ypred =  self.CheckField("Прогноз")            
             #Ytrue =  self.dataset["Истина"]
             #Ypred =  self.thread.result["Прогноз"]

             if(self.thread.result is None):
                  raise Exception("Отсутствует результат вычислений!")    

             if self.ui.cbModel.currentText() == "Классификация":
                  Ytrue =  self.CheckField("Р: Истина")
                  Ypred =  self.CheckField("Р: Прогноз")    
             elif self.ui.cbModel.currentText() == "Кластеризация":
                  if not "Истина" in self.dataset.columns:
                      raise Exception("Отсутствует столбец \"Истина\"!")
                  Ytrue =  self.CheckField("И: Истина")
                  Ypred =  self.CheckField("Р: Прогноз")    

             list = []
             list.clear()
             mat_con = confusion_matrix(Ytrue,Ypred)

             # Гуляев - новые метрики (+ для кластеризации)
             list.append("Accuracy: "+str(round(accuracy_score(Ytrue,Ypred),3)))
             list.append("Precision: "+str(round(precision_score(Ytrue,Ypred, average='weighted',zero_division=0),3)))
             list.append("Recall: "+str(round(recall_score(Ytrue,Ypred, average='weighted',zero_division=0),3)))
             list.append("Mean absolute error: "+str(round(mean_absolute_error(Ytrue,Ypred),3)))
             list.append("F1: "+str(round(f1_score(Ytrue,Ypred, average='weighted',zero_division=0),3)))
             list.append("----------Оценка производительности\n----------")
             list.append("Индекс Рэнда: "+str(round(metrics.adjusted_rand_score(Ytrue, Ypred), 3)))
             list.append("Взаимная\nинформация: "+str(round(metrics.adjusted_mutual_info_score(Ytrue, Ypred), 3)))
             list.append("Полнота: "+str(round(metrics.completeness_score(Ytrue, Ypred), 3)))
             list.append("V-мера: "+str(round(metrics.v_measure_score(Ytrue, Ypred), 3)))             

             self.figureMetrics.clear() # чистим график
             ax = self.figureMetrics.add_subplot()
             ax.set_title("Матрица неточности")            
             ax.set_xlabel("Predictions") # задаем название осей
             ax.set_ylabel("Actuals")             
             axx = ax.matshow(mat_con, cmap=plt.cm.YlOrRd, alpha=0.5)
             for m in range(mat_con.shape[0]):
                 for n in range(mat_con.shape[1]):
                    ax.text(x=m,y=n,s=mat_con[n, m], va='center', ha='center', size='xx-large')                                 
             self.figureMetrics.colorbar(axx)
             self.figureMetrics.tight_layout()
             self.canvasMetrics.draw()  # обновляем отрисовку графика             
             self.ui.tbMetrics.setText('\n'.join(list))
         except Exception as err:
            QMessageBox.about(self, "Ошибка!", "Вычислить метрики не удалось!\n"+str(err)) 
       

    def SwitchEnabledAll(self,flag):
         self.SwitchButtonsEnabled(flag)
         self.ui.cbModel.setEnabled(flag)
         self.ui.cbAlgorithm.setEnabled(flag)
         self.ui.cbData.setEnabled(flag)
         self.ui.cbTarget.setEnabled(flag)
         self.ui.cbY.setEnabled(flag)
         self.ui.cbX.setEnabled(flag)
         self.ui.cbZ.setEnabled(flag)
         self.ui.cbColor.setEnabled(flag)
         #self.ui.cbYtrue.setEnabled(flag)
         #self.ui.cbYpred.setEnabled(flag)
         self.ui.cbCluster1.setEnabled(flag)
         self.ui.cbCluster2.setEnabled(flag)
         self.ui.buttonOpenFile.setEnabled(flag)
         #self.ui.buttonReference.setEnabled(flag)

         if flag:
             self.ui.tbStatus.setText("Ожидаю...")
             self.ui.buttonLaunch.setText("Запустить")
         else:
             self.ui.tbStatus.setText("Происходят вычисления...")
             self.ui.buttonLaunch.setText("Остановить")            
             

    def GetX(self):
         data = []
         data.clear()
         #k = 0
         #a = self.ui.cbData.itemText(1)
         #for name in self.dataset.columns:
         #   k= k+1
         #   if self.itemChecked(k):
         #       data.append(name)
         #return self.dataset[data]
         for i in range(1,self.ui.cbData.count()):
             if self.itemChecked(i):
                 data.append(self.ui.cbData.itemText(i))

         return self.dataset[data]


    def GetY(self,y):
        df = y.drop_duplicates()
        t = y.copy()

        if df.dtype == np.int64:
            if df.sum() == (len(df)-1)*len(df)/2:
                return t
        a = 0
        k = 0
        for i in t.items():
            for j in df.items():
               if i[1]==j[1]:
                  t.iat[k] = a
                  break
               a=a+1
            k=k+1
            a = 0
        t = pd.to_numeric(y)
        return t


    def Launch(self):  
        self.ClearResult()
        self.FillXYZ()
        #self.FillMetrics()

        if  self.thread.isRunning():
            self.thread.terminate()
            self.SwitchEnabledAll(True)
            self.ui.buttonSwap.setEnabled(False)
            self.ui.buttonCopyToExcel.setEnabled(False)
            return
        
        countSelectColumns = 0  # считаем количество выбраных ячеек
       # for i in range(len(self.dataset.columns)):
        for i in range(1,self.ui.cbData.count()):
            if self.itemChecked(i):
                countSelectColumns=countSelectColumns+1
               
        if countSelectColumns==0:
            QMessageBox.about(self, "Ошибка!", "В поле Data должен быть выбран хотя бы одни столбец!")
            return
        
        if self.ui.cbTarget.currentText() == "" and self.ui.cbModel.currentText() == "Классификация":
             QMessageBox.about(self, "Ошибка!", "Поле Target не выбрано!")
             return
              
        # Готовим данные
        if self.flag == False:
            try:
             self.thread.table = self.GetX()
        
             if self.ui.cbModel.currentText()=="Классификация":
               self.thread.target = self.GetY(self.dataset[self.ui.cbTarget.currentText()])
        
             self.flag = True # данные собраны
            except Exception as err:      
                 QMessageBox.about(self, "Ошибка!", "Ошибка считывания данных выбранных столбцов.\n"+str(err))
                 return
        
        self.thread.parametrs = self.SetParametrs()
        self.SwitchEnabledAll(False)
        self.time = 0
        self.timer = QTimer(self)                                    
        self.timer.timeout.connect(self.showTime)
        self.timer.start(1)
        self.thread.start()


    def showTime(self):  
        self.time += 1
        self.settimer(self.time)

    def settimer(self, int):
        self.time = int
        self.ui.timeViewer.setText(str(self.time))
        self.ui.timeViewer.setAlignment(Qt.AlignCenter) # иначе текст снова будет слева

    def SetParametrs(self):
       parametrs = []

       if self.ui.cbAlgorithm.currentText() == "Nearest_Neighbors":
           parametrs.append(self.ui.spinBox.value())
           parametrs.append(self.ui.comboBox_7.currentText())
           parametrs.append(self.ui.comboBox_8.currentText())
           parametrs.append(self.ui.doubleSpinBox.value())
       elif self.ui.cbAlgorithm.currentText() == "Agglomerative":
           parametrs.append(self.ui.comboBox_9.currentText())
           parametrs.append(self.ui.spinBox_2.value())
           parametrs.append(self.ui.comboBox_10.currentText())
       elif self.thread.algorithm == "Affinity_Propagation":
           parametrs.append(self.ui.comboBox_11.currentText())
           parametrs.append(self.ui.spinBox_3.value())
           parametrs.append(self.ui.doubleSpinBox_2.value())
           parametrs.append(self.ui.doubleSpinBox_3.value())
           parametrs.append(self.ui.spinBox_10.value())
       elif self.thread.algorithm == "OPTICS":
           parametrs.append(self.ui.spinBox_4.value())
           parametrs.append(self.ui.doubleSpinBox_4.value())
           parametrs.append(self.ui.doubleSpinBox_5.value())
       elif self.thread.algorithm == "K_Means":
           parametrs.append(self.ui.spinBox_5.value())
           #parametrs.append(self.ui.comboBox_12.currentText())
           parametrs.append(self.ui.spinBox_6.value())
       elif self.thread.algorithm == "SGDClassifier":
           parametrs.append(self.ui.doubleSpinBox_6.value())
           parametrs.append(self.ui.spinBox_7.value())
           parametrs.append(self.ui.doubleSpinBox_7.value())
       elif self.thread.algorithm == "SVM":            
           parametrs.append(self.ui.doubleSpinBox_8.value())
           parametrs.append(self.ui.comboBox_13.currentText())
       elif self.thread.algorithm == "Gaussian_Process":
           parametrs.append(self.ui.doubleSpinBox_9.value())
           parametrs.append(self.ui.spinBox_8.value())
           parametrs.append(self.ui.spinBox_9.value())
       elif self.thread.algorithm == "Mean_Shift":
           parametrs.append(self.ui.MS_bd.value())
       elif self.thread.algorithm == "myBirch":
           parametrs.append(self.ui.Birch_nc.value())
       elif self.thread.algorithm == "Random_Forest":
           parametrs.append(self.ui.RF_ts.value())
           parametrs.append(self.ui.RF_md.value())
           parametrs.append(self.ui.RF_rs.value())
       elif self.thread.algorithm == "N_Byes":
           parametrs.append(self.ui.NB_ts.value())
           parametrs.append(self.ui.NB_rs.value())
       elif self.thread.algorithm == "Spectral":
           parametrs.append(self.ui.SCL_ncl.value())
           parametrs.append(self.ui.SCL_asgls.currentText())
           parametrs.append(self.ui.SCL_rs.value())
       elif self.thread.algorithm == "D_Tree":
           parametrs.append(self.ui.DT_RS.value())
       elif self.thread.algorithm == "DBScan":
           parametrs.append(self.ui.DBScan_eps.value())
           parametrs.append(self.ui.DBScan_ms.value())
       return parametrs 


     
    def CopyToExcel(self):
        path = QFileDialog.getSaveFileName(self, 'Выберите файл',filter = "(*.xlsx)")[0]

        if path!="":
            try:
               data = None

               if self.ui.cbModel.currentText()=="Кластеризация":
                  data = pd.concat([self.thread.table, self.thread.result], axis=1).copy()
               else:
                  data = self.thread.result.copy()

               #data = data.astype(str).apply(lambda x: x.str.replace(',', '.', regex=True))
               #data = data.apply(pd.to_numeric, errors='ignore')# coerce
               data.to_excel(path, index=False)

               QMessageBox.about(self, "Уведомление!", "Файл успешно сохранен!")
            except Exception as err:
               QMessageBox.about(self, "Ошибка!", "Сохранить файл не удалось!\n"+str(err))


    def FillSwap(self):
        countClusters = len(self.thread.result[self.thread.result.columns[-1]].drop_duplicates())
        self.ui.cbCluster1.addItem("")
        self.ui.cbCluster2.addItem("")
        for i in range(countClusters):
            self.ui.cbCluster1.addItem(str(i))
            self.ui.cbCluster2.addItem(str(i))


    def Swap(self):
         try:
             if self.ui.cbCluster1.currentText() == self.ui.cbCluster2.currentText():
              raise Exception("Кластеры должны быть разными!") 

             if self.ui.cbCluster1.currentText() == "":
              raise Exception("Кластер 1 не выбран!") 

             if self.ui.cbCluster2.currentText() == "":
              raise Exception("Кластер 2 не выбран!") 

             self.thread.result = self.thread.result.replace({self.thread.result.columns[-1]:{int(self.ui.cbCluster1.currentText()):int(self.ui.cbCluster2.currentText()), int(self.ui.cbCluster2.currentText()):int(self.ui.cbCluster1.currentText())}})

             self.figureChart.clear()
             self.canvasChart.draw()
         except Exception as err:
            QMessageBox.about(self, "Ошибка!", "Поменять кластеры не удалось!\n"+str(err)) 


    #def ShowReference(self):
    #        font = QFont()
    #        font.setFamily("Arial")
    #        font.setPointSize(15)
    #        dlg = QMessageBox(self)
    #        dlg.setWindowTitle("Руководство пользователя")
    #        dlg.setText("123")
    #        dlg.setFont(font)
    #        dlg.exec()

    def ConnectButtonsEvents(self):
        self.ui.cbModel.activated.connect(self.SwitchModel)
        self.ui.cbAlgorithm.activated.connect(self.SwitchAlgorithm)
        self.ui.buttonOpenFile.clicked.connect(self.OpenFile)
        self.ui.buttonLaunch.clicked.connect(self.Launch)
        self.ui.buttonCopyToExcel.clicked.connect(self.CopyToExcel)
       # self.ui.buttonReference.clicked.connect(self.ShowReference)
        self.ui.buttonDraw.clicked.connect(self.DrawChart)
        self.ui.buttonSwap.clicked.connect(self.Swap)
        self.ui.cbTarget.activated.connect(self.SwitchTarget)
        self.ui.buttonMetrics.clicked.connect(self.CalculateMetrics)

    # Заполняем comboBox перечислениями,если они есть для алгоритма
    def FillComboBox(self):
        for i in AffinityType:
            self.ui.comboBox_10.addItem(i.name)

        for i in LinkageType:
            self.ui.comboBox_9.addItem(i.name)

        for i in KNNAlgorithmType:
            self.ui.comboBox_7.addItem(i.name)

        for i in KNNWeightType:
            self.ui.comboBox_8.addItem(i.name)

        for i in Affinity:
            self.ui.comboBox_11.addItem(i.name)

        for i in Gamma:
            self.ui.comboBox_13.addItem(i.name)

        for i in AssignLabels:
            self.ui.SCL_asgls.addItem(i.name)


    def CreateCharts(self):
        self.figureChart = plt.figure()
        self.canvasChart = FigureCanvas(self.figureChart)

        self.figureMetrics = plt.figure()
        self.canvasMetrics = FigureCanvas(self.figureMetrics)

        self.toolbar = NavigationToolbar(self.canvasChart, self)
        self.ui.layoutCanvasChart.addWidget(self.toolbar)
        self.ui.layoutCanvasChart.addWidget(self.canvasChart)
        self.ui.layoutCanvasMetrics.addWidget(self.canvasMetrics)


    def SettingAnything(self):
        self.ui.container.hide()
        self.ui.tab.layout = QVBoxLayout() 
        self.ui.tbStatus.setText("Ожидаю...")


    def SettingComboBoxData(self):
        self.ui.cbData._changed = False 
        self.ui.cbData.hidePopup2 = self.ui.cbData.hidePopup
        self.ui.cbData.hidePopup = self.hidePopup.__get__(self.ui.cbData, QComboBox)
        self.ui.cbData.view().pressed.connect(self.handleItemPressed) 
        self.ui.cbData.currentTextChanged.connect(self.on_combobox_changed)


    def SetWindowSize(self):
        # устанавливаем окно по середине экрана и его размер равный половине разрешения экрана
        size = QApplication.primaryScreen().size()
        x = int(size.width()/2)
        y = int(size.height()/1.5)
        self.setGeometry(int(x/2),int(y/4),x,y);

    def ConfigureInterface(self):
        self.SetWindowSize()
        self.ConnectButtonsEvents()
        self.FillComboBox()         
        self.CreateCharts()
        self.SettingAnything()
        self.SettingComboBoxData()
        self.SwitchButtonsEnabled(False)
        self.SetHelp()


    def CheckField(self,cbBox):
        if cbBox.startswith("И:"):
               return self.dataset[cbBox.replace("И: ", "", 1)]
        elif cbBox.startswith("Р:"):
               return self.thread.result[cbBox.replace("Р: ", "", 1)]
        return None


    def DrawChart(self):
        try:       
           if self.ui.cbX.currentText()=="":
              raise Exception("Поле X не выбрано!") 

           if self.ui.cbY.currentText()=="":
              raise Exception("Поле Y не выбрано!") 
           
           if self.ui.cbColor.currentText()=="":
              raise Exception("Поле Color не выбрано!")   
          
           X = self.CheckField(self.ui.cbX.currentText())
           Y = self.CheckField(self.ui.cbY.currentText())
           Z = self.CheckField(self.ui.cbZ.currentText())
           Color = self.GetY(self.CheckField(self.ui.cbColor.currentText()))

           if Z is not None:
              Draw3DGraph(self.figureChart,X,Y,Z,Color)
           else:
              Draw2DGraph(self.figureChart,X,Y,Color)

           self.figureChart.tight_layout()
           self.canvasChart.draw()
        except Exception as err:
                self.figureChart.clear()
                self.canvasChart.draw()
                QMessageBox.about(self, "Ошибка!", "Отрисовать график не удалось!\n"+str(err))  


    def on_combobox_changed(self, value):
        self.ui.cbData.setCurrentText(" ")


    def hidePopup(self): 
        self.ui.cbData.setCurrentText(' ')
        if not self.ui.cbData._changed: 
            self.ui.cbData.hidePopup2() 
        self.ui.cbData._changed = False 
        

    def handleItemPressed(self,index):
        if index.row()!=0:
            item = self.ui.cbData.model().itemFromIndex(index) 
            if item.checkState() == Qt.Checked: 
                item.setCheckState(Qt.Unchecked) 
            else: 
                item.setCheckState(Qt.Checked) 
            self.ui.cbData._changed = True 
            self.flag = False
            
            
    def itemChecked(self, index): 
        item = self.ui.cbData.model().item(index, self.ui.cbData.modelColumn()) 
        return item.checkState() == Qt.Checked


if __name__ == "__main__":
    app = QApplication([])
    app.setStyle('Breeze')


    application = Canvas()    
    application.show()
    sys.exit(app.exec())




    #def FillTable(self,data,table):
    #      rowsCount = len(data)
    #      columnsCount = len(data.columns)
    #      
    #      table.setColumnCount(columnsCount)
    #      table.setRowCount(rowsCount)
    #      table.setHorizontalHeaderLabels(data.columns) 
    #
    #      for i in range(rowsCount):
    #          for j in range(columnsCount):
    #              table.setItem(i, j, QTableWidgetItem(str(data.iat[i,j])))