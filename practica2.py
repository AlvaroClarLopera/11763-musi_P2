import cv2
import logging
import pydicom
import copy as cp
import pandas as pd
import numpy as np
import corregistro
import segmentacion
from matplotlib import pyplot as plt

from tkinter import *
from tkinter import ttk
from tkinter import filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

global row,col,val
global val_corte
global imageDICOM
global directory
global filename
row = 0
col = 0
val = 0
import os
colorFondo = "#000040"
colorLletra = "#FFF"
scale = 1
offset = [0, 0]
current_scale = [offset[0], offset[1], scale, scale]

def load_images():
    global imageDICOM
    global val_corte
    global directory
    global filename
    val_corte = 0
    directory = "imagenes_dicom/"
    filename = filedialog.askopenfilename(initialdir=directory,title = "Select file")
    #filename = "000000.dcm"
    logging.info("Se ha cargado el archivo DICOM en "+str(filename))
    imageDICOM = pydicom.dcmread(filename)
    print(imageDICOM)
    if not hasattr(imageDICOM,"PixelSpacing"):
        imageDICOM.add_new([0x0028,0x0030],"DS",[1,1])
    #directory = "imagenes_dicom/"
    #filename = "40826324_s1_CT_FB_masked.dcm"
    #imageDICOM = pydicom.read_file(directory+filename)
    # directory = "imagenes_dicom/"
    # filename = "PMD8540804318002412548_s04_T1_REST_Frame_1__PCARDM1.dcm"
    # imageDICOM = pydicom.read_file(directory+filename)
    #print(imageDICOM)
    #print(pydicom.pixel_data_handlers)
    #print(imageDICOM.pixel_array)

    image = imageDICOM.pixel_array
    if len(imageDICOM.pixel_array.shape) > 2:
        image = imageDICOM.pixel_array[val_corte]
    else:
        image = imageDICOM.pixel_array
        # df = pd.DataFrame(imageDICOM.values())
        # #print(df[0])
        # df[0] = df[0].apply(
        #     lambda x: pydicom.dataelem.DataElement_from_raw(x) if isinstance(x, pydicom.dataelem.RawDataElement) else x)
        # df['name'] = df[0].apply(lambda x: x.name)
        # df['value'] = df[0].apply(lambda x: x.value)
        # df = df[['name', 'value']]
    #print(imageDICOM.values())


def quit():
    exit(1)

def view_DICOM_headers():
    global tree
    global filename
    global imageDICOM
    global names
    global values
    global firstView
    ventanaH = Tk()
    ventanaH.resizable(0, 0)
    ventanaH.title("Cabeceras DICOM")
    ventanaH.geometry("1000x640")
    ventanaH.configure(background=colorFondo)
    column_names = ["name","value"]
    #imageDICOM = pydicom.dcmread(filename)
    logging.info("Se han consultado las cabecaras DICOM del archivo "+str(filename))
    df = pd.DataFrame(columns=column_names)
    if firstView == True:
        firstView = False
        names = []
        values = []
        for v in imageDICOM.values():
            if type(v) == pydicom.dataelem.RawDataElement:
                values.append(pydicom.dataelem.DataElement_from_raw(v).value)
            else:
                values.append(v)
        for n in imageDICOM:
            names.append(n.name)
    #print(len(names))
    df.insert(0,"name",names,allow_duplicates=True)
    df.insert(1,"value",values,allow_duplicates=True)

    tree = ttk.Treeview(ventanaH, height=40)
    vsb = ttk.Scrollbar(ventanaH, orient="vertical", command=tree.yview)
    vsb.place(x=980, y=0, height=640)

    tree.configure(yscrollcommand=vsb.set)
    df_col = df.columns.values

    counter = len(df)
    tree["columns"] = ("name", "value")
    rowLabels = df.index.tolist()

    for x in range(len(df_col)):
        tree.column(df_col[x], width=400)
        tree.heading(df_col[x], text=df_col[x])
        for i in range(counter):
            if x == 0:
                tree.insert('', i, text=rowLabels[i], values=df.iloc[i, :].tolist())

    tree.pack()

    mainloop()

def callback(event):
    global row
    global col
    global val
    global dist
    global medirLong
    global r1,r2,c1,c2

    if medirLong == True:
        if dist == 0:
            r1 = h - event.y
            c1 = event.x
            canvas.get_tk_widget().create_oval(event.x - 5, h - event.y - 5, event.x + 5, h - event.y + 5, width=3,
                                               outline="green", tag="oval1")
            dist += 1

            Label(ventanaV, text=r1).place(x=180, y=160)

            Label(ventanaV, text=c1).place(x=180, y=180)
        elif dist == 1:
            r2 = h - event.y
            c2 = event.x
            canvas.get_tk_widget().create_oval(event.x - 5, h - event.y - 5, event.x + 5, h - event.y + 5, width=3,
                                               outline="green", tag="oval2")
            dist += 1

            Label(ventanaV, text=r2).place(x=180, y=240)

            Label(ventanaV, text=c2).place(x=180, y=260)
            if hasattr(imageDICOM, 'PixelSpacing'):
                rt = (r2 - r1) * float(imageDICOM.PixelSpacing[0])
                ct = (c2 - c1) * float(imageDICOM.PixelSpacing[1])
            else:
                rt = (r2 - r1)
                ct = (c2 - c1)
            d = np.hypot(rt,ct)
            d = np.round(d,decimals=3)

            Label(ventanaV, text=d).place(x=180, y=300)
        else:
            dist = 0
            canvas.get_tk_widget().delete("oval1")
            canvas.get_tk_widget().delete("oval2")
            Label(ventanaV, text="                            ", bg=colorFondo,
                  fg=colorLletra).place(x=180, y=160)
            Label(ventanaV, text="                            ", bg=colorFondo,
                  fg=colorLletra).place(x=180, y=180)
            Label(ventanaV, text="                            ", bg=colorFondo,
                  fg=colorLletra).place(x=180, y=240)
            Label(ventanaV, text="                            ", bg=colorFondo,
                  fg=colorLletra).place(x=180, y=260)
            Label(ventanaV, text="             ", bg=colorFondo).place(x=180, y=300)

    else:
        dist = 0

    val = image[h-event.y][event.x]
    row = h-event.y
    col = event.x
    Label(ventanaV, text="              ",bg=colorFondo).place(x=1100,y=10)
    Label(ventanaV, text="              ",bg=colorFondo).place(x=1100,y=60)
    Label(ventanaV, text="              ",bg=colorFondo).place(x=1100,y=110)

    Label(ventanaV, text=row).place(x=1100,y=10)

    Label(ventanaV,text=col).place(x=1100,y=60)

    Label(ventanaV,text=val).place(x=1100,y=110)

def activarMedLong():
    global medirLong
    if medirLong == True:
        Label(ventanaV, text="                    ", bg=colorFondo,
              fg=colorLletra).place(x=80, y=160)
        Label(ventanaV, text="                    ", bg=colorFondo,
              fg=colorLletra).place(x=180, y=160)

        Label(ventanaV, text="                            ", bg=colorFondo,
              fg=colorLletra).place(x=80, y=180)
        Label(ventanaV, text="                            ", bg=colorFondo,
              fg=colorLletra).place(x=180, y=180)

        Label(ventanaV, text="                    ", bg=colorFondo,
              fg=colorLletra).place(x=80, y=240)
        Label(ventanaV, text="                    ", bg=colorFondo,
              fg=colorLletra).place(x=180, y=240)

        Label(ventanaV, text="                            ", bg=colorFondo,
              fg=colorLletra).place(x=80, y=260)
        Label(ventanaV, text="                            ", bg=colorFondo,
              fg=colorLletra).place(x=180, y=260)
        Label(ventanaV, text="                                               ", bg=colorFondo,
              fg=colorLletra).place(x=60, y=300)
        Label(ventanaV, text="                          ", bg=colorFondo).place(x=180, y=300)
        medirLong = False
    else:
        medirLong = True
        Label(ventanaV, text="Fila pixel 1", bg=colorFondo,
              fg=colorLletra).place(x=80, y=160)

        Label(ventanaV, text="Columna pixel 1", bg=colorFondo,
              fg=colorLletra).place(x=80, y=180)
        Label(ventanaV, text="Fila pixel 2", bg=colorFondo,
              fg=colorLletra).place(x=80, y=240)
        Label(ventanaV, text="Columna pixel 2", bg=colorFondo,
              fg=colorLletra).place(x=80, y=260)
        Label(ventanaV, text="Longitud entre pixeles", bg=colorFondo,
              fg=colorLletra).place(x=60, y=300)

def cambiarCorte3V(valor,corte):
    global val_corteA
    global val_corteS
    global val_corteC
    global corteA
    global oorteC
    global corteS
    global image
    global filename
    global directory
    # print(imageDICOM.pixel_array.shape)

    if valor == -1: #anterior corte
        if corte == 0:
            val_corteA -= 1
            if val_corteA < 0:
                val_corteA = imageF.shape[0] - 1
        elif corte == 1:
            val_corteC -= 1
            if val_corteC < 0:
                val_corteC = imageF.shape[1] - 1
        else:
            val_corteS -= 1
            if val_corteS < 0:
                val_corteS = imageF.shape[2] - 1
    else: #siguiente corte
        if corte == 0:
            val_corteA += 1
            if val_corteA > imageF.shape[0]:
                val_corteA = 0
        elif corte == 1:
            val_corteC += 1
            if val_corteC > imageF.shape[1]:
                val_corteC = 0
        else:
            val_corteS += 1
            if val_corteS > imageF.shape[2]:
                val_corteS = 0
    corteA = imageF[val_corteA, :, :]
    corteC = imageF[:, val_corteC, :]
    corteS = imageF[:, :, val_corteS]
    my_dpi = 100
    pixel_len_mm = [imageDICOM.SliceThickness, imageDICOM.PixelSpacing[0],
                    imageDICOM.PixelSpacing[1]]  # slice thickness, pixel spacing 0 1
    plt.subplot(2, 2, 1)  # axial
    plt.imshow(corteA, cmap=plt.cm.get_cmap('bone'), aspect=pixel_len_mm[1] / pixel_len_mm[2])
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)
    plt.subplot(2, 2, 2)  # coronal
    plt.imshow(corteC, cmap=plt.cm.get_cmap('bone'), aspect=pixel_len_mm[0] / pixel_len_mm[2])
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)
    plt.subplot(2, 2, 3)  # sagital
    plt.imshow(corteS, cmap=plt.cm.get_cmap('bone'), aspect=pixel_len_mm[0] / pixel_len_mm[1])
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)


    canvas3V.figure = fig3
    canvas3V.draw()
    frame3.pack(side=TOP, fill=X)
    canvas.get_tk_widget().pack(side=TOP, expand=1)

def ver3Cortes():
    global fig3
    global frame3
    global ventana3V
    global canvas3V #canvas para los 3 cortes
    global canvasA #axial
    global canvasS #sagital
    global canvasC #coronal
    global val_corteA
    global val_corteS
    global val_corteC
    ventana3V = Tk()
    ventana3V.resizable(0, 0)
    ventana3V.title("Visualizador por cortes")
    ventana3V.geometry("1280x720")
    ventana3V.configure(background=colorFondo)
    my_dpi = 100
    val_corteA = 0
    val_corteS = 0
    val_corteC = 0
    global imageF
    global corteA
    global corteS
    global corteC
    imageF = np.flip(imageDICOM.pixel_array,axis=0)
    corteA = imageF[val_corteA,:,:]
    corteC = imageF[:,val_corteC,:]
    corteS = imageF[:,:,val_corteS]
    fig3 = plt.figure(figsize=(600/my_dpi, 600/my_dpi), dpi=my_dpi, facecolor=colorFondo)
    pixel_len_mm = [imageDICOM.SliceThickness, imageDICOM.PixelSpacing[0], imageDICOM.PixelSpacing[1]] #slice thickness, pixel spacing 0 1
    plt.subplot(2, 2, 1) #axial
    plt.imshow(corteA, cmap=plt.cm.get_cmap('bone'),aspect=pixel_len_mm[1]/pixel_len_mm[2])
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)
    plt.subplot(2, 2, 2) #coronal
    plt.imshow(corteC, cmap=plt.cm.get_cmap('bone'),aspect=pixel_len_mm[0]/pixel_len_mm[2])
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)
    plt.subplot(2, 2, 3) #sagital
    plt.imshow(corteS, cmap=plt.cm.get_cmap('bone'),aspect=pixel_len_mm[0]/pixel_len_mm[1])
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)

    frame3 = Frame(ventana3V)

    canvas3V = FigureCanvasTkAgg(fig3, master=ventana3V)
    canvas3V.draw()
    frame3.pack(side=TOP, fill=X)
    canvas3V.get_tk_widget().pack(side=TOP, expand=1)

    toolbar = NavigationToolbar2Tk(canvas3V, frame3)
    toolbar.config(background=colorFondo)
    toolbar._message_label.config(background=colorFondo)
    toolbar.update()

    lArrow = PhotoImage(file="igu_data/left_arrow.png", master=ventana3V)
    rArrow = PhotoImage(file="igu_data/right_arrow.png", master=ventana3V)
    #axial
    Label(ventana3V, text="Cambiar corte axial", bg="#000000", fg="white").place(x=1150,y=70)
    Button(ventana3V, width=40, height=40, image=lArrow, command=lambda: cambiarCorte3V(-1,0), fg="white").place(x=1150, y=100)
    Button(ventana3V, width=40, height=40, image=rArrow, command=lambda: cambiarCorte3V(1,0), fg="white").place(x=1200, y=100)
    #coronal
    Label(ventana3V, text="Cambiar corte coronal", bg="#000000", fg="white").place(x=1150,y=170)
    Button(ventana3V, width=40, height=40, image=lArrow, command=lambda: cambiarCorte3V(-1,1), fg="white").place(x=1150, y=200)
    Button(ventana3V, width=40, height=40, image=rArrow, command=lambda: cambiarCorte3V(1,1), fg="white").place(x=1200, y=200)
    #sagital
    Label(ventana3V, text="Cambiar corte sagital", bg="#000000", fg="white").place(x=1150,y=270)
    Button(ventana3V, width=40, height=40, image=lArrow, command=lambda: cambiarCorte3V(-1,2), fg="white").place(x=1150, y=300)
    Button(ventana3V, width=40, height=40, image=rArrow, command=lambda: cambiarCorte3V(1,2), fg="white").place(x=1200, y=300)

    mainloop()




def createViewerInterface():
    global firstView
    global ventanaV
    global fig
    global imageDICOM
    global frame
    global dist
    global medirLong
    dist = 0
    medirLong = False
    ventanaV = Tk()
    firstView = True
    ventanaV.resizable(0, 0)
    ventanaV.title("Practica 1")
    ventanaV.geometry("1280x720")
    ventanaV.configure(background=colorFondo)
    global image
    if len(imageDICOM.pixel_array.shape) > 2:
        image = imageDICOM.pixel_array[0]
    else:
        image = imageDICOM.pixel_array
    #image = cv2.imread('mandril_color.tif')

    my_dpi = 100  # Good default - doesn't really matter

    # Size of output in pixels
    global w,h
    w = image.shape[0]
    h = image.shape[1]
    fig = plt.figure(figsize=(w/my_dpi, h/my_dpi), dpi=my_dpi)
    plt.imshow(image)  # later use a.set_data(new_data)
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)

    frame = Frame(ventanaV)


    left_arrow = cv2.imread("igu_data/left_arrow.png")
    #left_arrow = cv2.resize(left_arrow,(40,40), interpolation=cv2.INTER_AREA)
    right_arrow = cv2.imread("igu_data/right_arrow.png")
    #right_arrow = cv2.resize(right_arrow,(40,40), interpolation=cv2.INTER_AREA)
    #cv2.imwrite("igu_data/left_arrow.png",left_arrow)
    #cv2.imwrite("igu_data/right_arrow.png",right_arrow)

    # a tk.DrawingArea
    global canvas
    canvas = FigureCanvasTkAgg(fig, master=ventanaV)
    canvas.draw()
    frame.pack(side=TOP, fill=X)
    canvas.get_tk_widget().pack(side=TOP, expand=1)

    toolbar = NavigationToolbar2Tk(canvas, frame)
    toolbar.config(background=colorFondo)
    toolbar._message_label.config(background=colorFondo)
    toolbar.update()

    fig.canvas.callbacks.connect('button_press_event', callback)

    Label(ventanaV, text="Fila", bg=colorFondo,
          fg=colorLletra).place(x=1000, y=10)

    Label(ventanaV, text="Columna", bg=colorFondo,
          fg=colorLletra).place(x=1000, y=60)

    Label(ventanaV, text="Valor del pixel", bg=colorFondo,
          fg=colorLletra).place(x=1000, y=110)

    scale = Scale(ventanaV, from_=0, to=100, orient=HORIZONTAL, showvalue = 50,command=ajustarContraste)
    scale.place(x=1000,y=250)
    scale.set(50)
    Label(ventanaV, text="Ajustar contraste", bg="#000000", fg="white").place(x=1000, y=220)

    Button(ventanaV, text="Medir longitud entre dos píxeles", command=activarMedLong, bg="#000000", fg="white").place(x=100, y=100)

    Button(ventanaV, text="Ver cabeceras DICOM", command=view_DICOM_headers, bg="#000000", fg="white").place(x=470, y=20)
    Button(ventanaV, text="Visualizar diferentes cortes (3D)", command=ver3Cortes, bg="#000000", fg="white").place(x=1000, y=300)
    Button(ventanaV, text="Realizar corregistro de imagenes", command=corregistro.visualizadorDoble_Fusion
           ,bg="#000000", fg="white").place(x=1000, y=350)
    Button(ventanaV, text="Realizar segmentación de imágenes", command=segmentacion.visualizadorSegmentacion,
           bg="#000000", fg="white").place(x=1000, y=400)
    Button(ventanaV, text="Salir", command=quit, bg="#000000", fg="white").place(x=780, y=20)

    lArrow = PhotoImage(file="igu_data/left_arrow.png", master=ventanaV)
    rArrow = PhotoImage(file="igu_data/right_arrow.png", master=ventanaV)

    Button(ventanaV, width=40, height=40, image=lArrow, command=lambda :cambiarCorte(-1),fg="white").place(x=600, y=50)
    Button(ventanaV, width=40, height=40, image=rArrow, command=lambda :cambiarCorte(1),fg="white").place(x=650, y=50)
    mainloop()

def createMainInterface():
    logging.basicConfig(filename='app.log', filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)
    logging.info("Se ha iniciado una nueva sesión")
    global filename
    filename = ""
    colorFondo = "#000040"
    ventana = Tk()
    ventana.resizable(0, 0)
    ventana.title("Practica 1")
    ventana.geometry("400x250")
    ventana.configure(background=colorFondo)
    Label(ventana, text="Visualizador \n de imágenes médicas DICOM.",font=(None, 20), bg=colorFondo, fg=colorLletra).place(x=10, y=10)
    Label(ventana, text="Corregistro y segmentación \n de imágenes DICOM.", font=(None, 20), bg=colorFondo, fg=colorLletra).place(x=10, y=80)
    Button(ventana, text="Cargar imagen", command=load_images, bg="#000000", fg="white").place(x=10, y=200)
    Button(ventana, text="Visualizar", command=createViewerInterface, bg="#000000", fg="white").place(x=160, y=200)
    Button(ventana, text="Salir", command=quit, bg="#000000", fg="white").place(x=310, y=200)
    mainloop()

def ajustarContraste(val):
    global w, h
    global image
    global imageC
    my_dpi = 100  # Good default - doesn't really matter
    vali = int(val)
    alpha = vali / 50

    #print("Alpha = "+str(alpha))
    #print("Beta = "+str(beta))
    #imageC = cp.deepcopy(image)
    # print("Por defecto: "+str(image[196][260]))
    max_act = np.amax(image)
    min_act = np.amin(image)
    # print("Max: "+str(max_act))
    # print("Min: "+str(min_act))
    #print("Alpha: "+str(alpha))
    newmax = int(max_act* alpha * 30)
    if min_act < 0:
        newmin = int(min_act * alpha * 30)
        newmin = int(min_act * alpha * 30)
    else:
        newmin = int(min_act / (alpha * 30 + 0.000001))

    imageC = cp.deepcopy(image)
    # print("newmax: "+str(newmax))
    # print("newmin: "+str(newmin))
    imageC[:,:] = (((newmax - newmin) * ((image[:,:] - min_act) / (max_act - min_act))) + newmin)
    # print(imageC[196][260])
    logging.info("Se ha ajustado el contraste a un valor alpha "+str(alpha))
    w = imageC.shape[0]
    h = imageC.shape[1]
    plt.imshow(imageC,cmap=plt.cm.bone)  # later use a.set_data(new_data)
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)


    # a tk.DrawingArea
    canvas.figure = fig
    canvas.draw()
    frame.pack(side=TOP, fill=X)
    canvas.get_tk_widget().pack(side=TOP, expand=1)

def cambiarCorte(valor):
    global val_corte
    global image
    global firstView
    global filename
    global directory
    global imageDICOM
    # print(imageDICOM.pixel_array.shape)
    if len(imageDICOM.pixel_array.shape) > 2:
        if valor == -1: #anterior corte
            val_corte -= 1
            if val_corte < 0:
                val_corte = imageDICOM.pixel_array.shape[0] - 1
        else: #siguiente corte
            val_corte += 1
            if val_corte > imageDICOM.pixel_array.shape[0] - 1:
                val_corte = 0

        imagep = imageDICOM.pixel_array[val_corte]
        print(imagep.shape)
        image = cp.deepcopy(imagep)
    else:
        if "imagenes_dicom/0-27993/" or "CT_Lung/" or "RM_Brain_3D-SPGR/" in filename:

            if valor == -1: #anterior corte
                val_corte -= 1
                if val_corte < 0:
                    val_corte = 117
            else: #siguiente corte
                val_corte += 1
                if val_corte > 117:
                    val_corte = 0
            cont = 0
            if "imagenes_dicom/0-27993/" in filename:
                directory = "imagenes_dicom/0-27993/"
            if "CT_Lung/" in filename:
                directory = "P2 - DICOM/CT_Lung/"
            if "RM_Brain_3D-SPGR/" in filename:
                directory = "P2 - DICOM/RM_Brain_3D-SPGR/"
            for fname in os.listdir(directory):
                print(directory+fname)
                imageDICOM = pydicom.dcmread(directory+fname)
                print(fname)
                if cont == val_corte:
                    break
                cont += 1
            imagep = imageDICOM.pixel_array
            firstView = True
            image = cp.deepcopy(imagep)
    logging.info("Se ha cambiado de corte. Actualmente se visualiza el corte nº "+str(val_corte+1))
    plt.imshow(image, cmap=plt.cm.bone)  # later use a.set_data(new_data)
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)

    # a tk.DrawingArea
    canvas.figure = fig
    canvas.draw()
    frame.pack(side=TOP, fill=X)
    canvas.get_tk_widget().pack(side=TOP, expand=1)

def sortSlices():
    files = []
    directory = "P2 - DICOM/RM_Brain_3D-SPGR/"
    for fname in os.listdir(directory):
        files.append(pydicom.dcmread(directory+fname))
    slices = []
    skipcount = 0
    for f in files:
        if hasattr(f, 'SliceLocation'):
            slices.append(f)
        else:
            skipcount = skipcount + 1

    print("skipped, no SliceLocation: {}".format(skipcount))

    # ensure they are in the correct order
    slices = sorted(slices, key=lambda s: s.SliceLocation)
    i = 0
    directory = "P2 - DICOM/RM_Brain_3D-SPGR_Sorted/"
    for l in slices:
        print(l.SliceLocation)
        pydicom.dcmwrite(directory+str(i)+".dcm",l)
        i += 1
createMainInterface()
#sortSlices()