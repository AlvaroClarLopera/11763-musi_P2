import numpy as np
import cv2
import pydicom
import pyelastix
import SimpleITK
#import itk
import copy as cp
import pandas as pd
from matplotlib import pyplot as plt
from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


colorFondo = "#000040"
colorLletra = "#FFF"
my_dpi = 100

global imageDICOM1
global imageDICOM2
global n_markers

def visualizadorSegmentacion():
    global figS
    global frameS
    global ventanaS
    global canvasS
    ventanaS= Tk()
    ventanaS.resizable(0, 0)
    ventanaS.title("Herramienta de corregistro")
    ventanaS.geometry("1280x720")
    ventanaS.configure(background=colorFondo)

    figS = plt.figure(figsize=(2000/my_dpi, 512/my_dpi), dpi=my_dpi, facecolor=colorFondo)
    #figS, axs = plt.subplots(1, 2, figsize=(60, 60))

    #pixel_len_mm = [imageDICOM.SliceThickness, imageDICOM.PixelSpacing[0], imageDICOM.PixelSpacing[1]] #slice thickness, pixel spacing 0 1

    Button(ventanaS, text="Cargar imagen a segmentar", command=loadImage, bg="#000000", fg="white").place(x=470, y=20)
    Button(ventanaS, text="Segmentar imagen",command=segmentar, bg="#000000", fg="white").place(x=800, y=20)

    plt.subplot(1, 2, 1)  # axial
    #plt.imshow(corteA, cmap=plt.cm.get_cmap('bone'), aspect=pixel_len_mm[1] / pixel_len_mm[2])
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)

    plt.subplot(1, 2, 2)  # axial
    # plt.imshow(corteA, cmap=plt.cm.get_cmap('bone'), aspect=pixel_len_mm[1] / pixel_len_mm[2])
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)

    frameS = Frame(ventanaS)

    canvasS = FigureCanvasTkAgg(figS, master=ventanaS)
    canvasS.draw()
    frameS.pack(side=BOTTOM, fill=X)
    canvasS.get_tk_widget().pack(side=BOTTOM, expand=1)

    figS.canvas.callbacks.connect('button_press_event', mark_regions_of_interest)



def loadImage():
    global imageDICOM1
    global image1
    global n_markers
    global markers
    n_markers = 0
    canvasS.get_tk_widget().delete("oval1")

    directory = "imagenes_dicom/"
    filename = filedialog.askopenfilename(initialdir=directory, title="Select file")
    imageDICOM1 = pydicom.dcmread(filename)
    if len(imageDICOM1.pixel_array.shape) > 2:
        image1 = imageDICOM1.pixel_array[0]
    else:
        image1 = imageDICOM1.pixel_array
    markers = np.zeros((image1.shape), dtype=np.int32)

    plt.subplot(1, 2, 1)  # axial
    plt.imshow(image1, cmap=plt.cm.get_cmap('bone'))  # aspect=pixel_len_mm[1] / pixel_len_mm[2]
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0.5)


    canvasS.figure = figS
    canvasS.draw()
    frameS.pack(side=TOP, fill=X)
    canvasS.get_tk_widget().pack(side=TOP, expand=1)

def segmentar():
    global markers
    '''
    cv2.imshow("res",image1_8U)
    cv2.waitKey(0)
    ret, thresh = cv2.threshold(image1_8U, 50, 255, cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    print("markers size: "+str(markers.shape))
    print("markers: "+str(np.unique(markers)))
    print("markers: "+str(markers))
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    image1_8U3 = cv2.cvtColor(image1_8U, cv2.COLOR_GRAY2BGR)
    '''
    image1_8U = cv2.convertScaleAbs(image1)
    image1_8U3 = cv2.cvtColor(image1_8U, cv2.COLOR_GRAY2BGR)
    #markers += markers
    markers = cv2.watershed(image1_8U3, markers)
    image1_8U3[markers == -1] = [255, 0, 0]


    plt.subplot(1, 2, 2)  # axial
    plt.imshow(image1_8U3, cmap=plt.cm.get_cmap('bone'))  # aspect=pixel_len_mm[1] / pixel_len_mm[2]
    plt.gca().set_axis_off()
    plt.margins(0, 0)

    canvasS.figure = figS
    canvasS.draw()
    frameS.pack(side=TOP, fill=X)

    canvasS.get_tk_widget().pack(side=TOP, expand=1)
    markers8U = cv2.convertScaleAbs(markers)
    markersJet = cv2.applyColorMap(markers8U, cv2.COLORMAP_JET);
    cv2.imshow("Resultado",markersJet)
    cv2.waitKey(0)

def mark_regions_of_interest(event):
    global w, h
    global n_markers
    global markers
    w = image1.shape[0]
    h = image1.shape[1]
    r = h - event.y
    c = event.x
    markers[r][c] = n_markers + 1
    n_markers += 1




    canvasS.get_tk_widget().create_oval(event.x - 5, h - event.y - 5, event.x + 5, h - event.y + 5, width=3,
                                       outline="green", tag="oval1")