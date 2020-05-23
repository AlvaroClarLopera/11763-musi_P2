import numpy as np
import os
import cv2
import pydicom
import statistics
import pyelastix
from scipy import ndimage
import SimpleITK as itk
#import itk
from skimage.morphology import disk
from skimage.filters import rank
from skimage import feature
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from skimage import filters
from scipy import ndimage as ndi
from skimage.segmentation import watershed as ws
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
val_corte = 0

def visualizadorSegmentacion():
    global figS
    global frameS
    global ventanaS
    global canvasS
    global segmented
    segmented = False
    ventanaS= Tk()
    ventanaS.resizable(0, 0)
    ventanaS.title("Herramienta de segmentación")
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
    figS.canvas.callbacks.connect('button_press_event', show_region_features)



def loadImage():
    global imageDICOM1
    global image1
    global val_corte
    global filename
    global n_markers
    global markers
    global imageITK1
    global segmented
    segmented = False
    n_markers = 0
    Label(ventanaS, text="            ", bg=colorFondo, fg="white").place(x=600, y=80)
    Label(ventanaS, text="                                   ", bg=colorFondo, fg="white").place(x=600, y=160)
    Label(ventanaS, text="           ", bg=colorFondo, fg="white").place(x=650, y=80)
    Label(ventanaS, text="           ", bg=colorFondo, fg="white").place(x=650, y=120)
    Label(ventanaS, text="           ", bg=colorFondo, fg="white").place(x=650, y=160)
    Label(ventanaS, text="           ", bg=colorFondo, fg="white").place(x=650, y=200)
    canvasS.get_tk_widget().delete("oval1")
    directory = "imagenes_dicom/"
    filename = filedialog.askopenfilename(initialdir=directory, title="Select file")
    imageITK1 = itk.ReadImage(filename, itk.sitkFloat32)
    imageITK1 = itk.Extract(imageITK1, (imageITK1.GetWidth(), imageITK1.GetHeight(), 0), (0, 0, 0))
    imageDICOM1 = pydicom.dcmread(filename)
    nameF = filename.rsplit('/', 1)
    nameF = nameF[1].rsplit('.', 1)

    val_corte = int(nameF[0])
    print(val_corte)
    if len(imageDICOM1.pixel_array.shape) > 2:
        image1 = imageDICOM1.pixel_array[val_corte]
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

    lArrow = PhotoImage(file="igu_data/left_arrow.png", master=ventanaS)
    rArrow = PhotoImage(file="igu_data/right_arrow.png", master=ventanaS)
    Button(ventanaS, width=30, height=30, image=lArrow, command=lambda: cambiarCorte(-1)).place(
        x=200, y=40)
    Button(ventanaS, width=30, height=30, image=rArrow, command=lambda: cambiarCorte(1)).place(
        x=250, y=40)
    mainloop()

def cambiarCorte(valor):
    global val_corte
    global image
    global firstView
    global markers
    global filename
    global segmented
    global directory
    global imageDICOM1
    global image1
    segmented = False
    canvasS.get_tk_widget().delete("oval1")
    # print(imageDICOM.pixel_array.shape)
    if len(imageDICOM1.pixel_array.shape) > 2:
        if valor == -1: #anterior corte
            val_corte -= 1
            if val_corte < 0:
                val_corte = imageDICOM1.pixel_array.shape[0] - 1
        else: #siguiente corte
            val_corte += 1
            if val_corte > imageDICOM1.pixel_array.shape[0] - 1:
                val_corte = 0

        imagep = imageDICOM1.pixel_array[val_corte]
        print(np.unique(imagep))
        print(val_corte)
        image1 = cp.deepcopy(imagep)

    else:
        if "imagenes_dicom/0-27993/" or "CT_Lung/" or "RM_Brain_3D-SPGR/" in filename:
            if "imagenes_dicom/0-27993/" in filename:
                directory = "imagenes_dicom/0-27993/"
            if "CT_Lung/" in filename:
                directory = "P2 - DICOM/CT_Lung/"
            if "RM_Brain_3D-SPGR/" in filename:
                directory = "P2 - DICOM/RM_Brain_3D-SPGR/"
            if valor == -1: #anterior corte
                val_corte -= 1
                if val_corte < 0:
                    val_corte = len(os.listdir(directory))
            else: #siguiente corte
                val_corte += 1
                if val_corte > len(os.listdir(directory)):
                    val_corte = 0
            cont = 0

            for fname in os.listdir(directory):
                print(directory+fname)
                imageDICOM1 = pydicom.dcmread(directory+fname)
                print(fname)
                if cont == val_corte:
                    break
                cont += 1
            imagep = imageDICOM1.pixel_array
            firstView = True
            image1 = cp.deepcopy(imagep)
            markers = np.zeros((image1.shape), dtype=np.int32)

    #logging.info("Se ha cambiado de corte. Actualmente se visualiza el corte nº "+str(val_corte+1))
    plt.subplot(1, 2, 1)  # axial
    plt.imshow(image1, cmap=plt.cm.bone)  # later use a.set_data(new_data)
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0.5)

    # a tk.DrawingArea
    canvasS.figure = figS
    canvasS.draw()
    frameS.pack(side=TOP, fill=X)
    canvasS.get_tk_widget().pack(side=TOP, expand=1)

def segmentar():
    global markers
    global image1
    global final_image
    global segmented
    segmented = True
    grad_img1 = filters.sobel(image1)

    #grad_img1_bw = cv2.convertScaleAbs(grad_img1)
    #(th, grad_img1_bw) = cv2.threshold(grad_img1, 127, 255, cv2.THRESH_BINARY)
    distance = ndi.distance_transform_edt(image1)

    #grad_img1 = feature.canny(image1, sigma=0.5)
    image1_seg = cp.deepcopy(image1)
    #markers += markers
    #mask = image1 > np.max()
    markers = ws(grad_img1, markers, watershed_line=True)
    #markers = cv2.watershed(image1, markers)
    #grad_img1 = itk.GradientMagnitude(imageITK1)
    #markers = itk.MorphologicalWatershedFromMarkers(grad_img1, markers, markWatershedLine=True, fullyConnected=False)
    #image1_seg[markers == -1] = [255, 0, 0]

    superpose_regions()
    plt.subplot(1, 2, 2)  # axial
    plt.imshow(markers, cmap=plt.cm.nipy_spectral)  # aspect=pixel_len_mm[1] / pixel_len_mm[2]
    #plt.imshow(image1, cmap=plt.cm.get_cmap("bone"))
    plt.gca().set_axis_off()
    plt.margins(0, 0)

    canvasS.figure = figS
    canvasS.draw()
    frameS.pack(side=TOP, fill=X)

    canvasS.get_tk_widget().pack(side=TOP, expand=1)

    '''
    markers8U = cv2.convertScaleAbs(markers)
    markersJet = cv2.applyColorMap(markers8U, cv2.COLORMAP_JET);
    cv2.imshow("Resultado",markersJet)
    cv2.waitKey(0)
    '''
def superpose_regions():
    global n_markers
    global markers
    global final_image
    global avgLabel
    global stdevLabel
    final_image = cp.deepcopy(markers)
    for nm in range(n_markers-1, n_markers):
        for i in range(0,markers.shape[0]):
            for j in range(0,markers.shape[1]):
                if nm == markers[i][j]:
                    final_image[i][j] = image1[i][j]
    Label(ventanaS, text="Media", bg="#000000", fg="white").place(x=550, y=80)
    Label(ventanaS, text="Desviación típica", bg="#000000", fg="white").place(x=550, y=120)
    Label(ventanaS, text="Máximo", bg="#000000", fg="white").place(x=550, y=160)
    Label(ventanaS, text="Mediana", bg="#000000", fg="white").place(x=550, y=200)
    Label(ventanaS, text="Total de pixeles ",bg="#000000", fg="white").place(x=550, y=240)

def show_region_features(event):
    global image1
    l_pos = 1000
    r_pos = -1
    u_pos = 1000
    d_pos = -1
    list_roi_pixels = []
    Label(ventanaS, text="                 ", bg=colorFondo).place(x=650, y=80)
    Label(ventanaS, text="                 ", bg=colorFondo).place(x=650, y=120)
    Label(ventanaS, text="                 ", bg=colorFondo).place(x=650, y=160)
    Label(ventanaS, text="                 ", bg=colorFondo).place(x=650, y=200)
    Label(ventanaS, text="                 ", bg=colorFondo).place(x=650, y=240)
    if segmented:
        r_roi = h - event.y
        c_roi = int(event.x - image1.shape[1] - (image1.shape[1] / 2))
        # print(r_roi)
        # print(c_roi)
        n_roi = markers[r_roi][c_roi]
        canvasS.get_tk_widget().delete("oval2")
        canvasS.get_tk_widget().create_oval(event.x - 5, h - event.y - 5, event.x + 5, h - event.y + 5, width=3,
                                            outline="white", tag="oval2")
        pixel_count = 0
        for i in range(0, image1.shape[0]):
            for j in range(0, image1.shape[1]):

                if markers[i][j] == n_roi:
                    pixel_count += 1
                    list_roi_pixels.append(image1[i][j]*1.0)
                    if i < u_pos:
                        u_pos = i
                    if i > d_pos:
                        d_pos = i
                    if j < l_pos:
                        l_pos = j
                    if j > r_pos:
                        r_pos = j
        print("left "+str(l_pos))
        print("right "+str(r_pos))
        print("up "+str(u_pos))
        print("down "+str(d_pos))
        img_roi = image1[u_pos:d_pos+1,l_pos:r_pos+1]
        #kernel = np.real(gabor_kernel(0.15, theta=4 / 4. * np.pi,sigma_x=3, sigma_y=3))
        #res = ndimage.convolve(img_roi, kernel, mode='reflect')
        g_filters = create_filter_bank()
        #cv2.imshow("res",res)
        #cv2.waitKey(0)


        mean = statistics.mean(list_roi_pixels)
        stdev = statistics.stdev(list_roi_pixels)
        median = statistics.median(list_roi_pixels)

        mean = np.round(mean, 3)
        stdev = np.round(stdev, 3)
        max_val = np.max(list_roi_pixels)
        Label(ventanaS, text=mean).place(x=650, y=80)
        Label(ventanaS, text=stdev).place(x=650, y=120)
        Label(ventanaS, text=max_val).place(x=650, y=160)
        Label(ventanaS, text=median).place(x=650, y=200)
        Label(ventanaS, text=pixel_count).place(x=650, y=240)

        figFilt, axs = plt.subplots(2, 4)
        axs = [a for ax in axs for a in ax]
        figFilt.suptitle('GABOR FILTERS')
        [ax.imshow(apply_filter(img_roi, k)) for k, ax in zip(g_filters, axs)]
        figFilt.show()

def create_filter_bank():
    """ Adapted from skimage doc. """
    kernels = []
    for theta in range(0, 2):
        theta = theta / 2. * np.pi
        for sigma in (3, 5):
            for frequency in (0.10, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)
    print(len(kernels))
    return kernels

def apply_filter(image, kernel):
    return ndimage.convolve(image, kernel, mode='reflect')

def mark_regions_of_interest(event):

    global w, h
    global n_markers
    global markers
    if not segmented:
        w = image1.shape[0]
        h = image1.shape[1]
        r = h - event.y
        c = event.x
        markers[r][c] = n_markers + 1
        n_markers += 1
        canvasS.get_tk_widget().create_oval(event.x - 5, h - event.y - 5, event.x + 5, h - event.y + 5, width=3,
                                            outline="green", tag="oval1")