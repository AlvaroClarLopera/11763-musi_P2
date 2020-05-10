import numpy as np
import cv2
import pydicom
import PIL
import os
import SimpleITK as itk
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

def visualizadorDoble_Fusion():
    global figD
    global frameD
    global ventanaD
    global canvasD
    global copy_img1
    global copy_img2
    ventanaD = Tk()
    ventanaD.resizable(0, 0)
    ventanaD.title("Herramienta de corregistro")
    ventanaD.geometry("1280x720")
    ventanaD.configure(background=colorFondo)

    figD = plt.figure(figsize=(600/my_dpi, 600/my_dpi), dpi=my_dpi, facecolor=colorFondo)
    #pixel_len_mm = [imageDICOM.SliceThickness, imageDICOM.PixelSpacing[0], imageDICOM.PixelSpacing[1]] #slice thickness, pixel spacing 0 1

    Button(ventanaD, text="Cargar imagen 1", command=lambda: loadImage(1), bg="#000000", fg="white").place(x=470, y=20)
    Button(ventanaD, text="Cargar imagen 2", command=lambda: loadImage(2), bg="#000000", fg="white").place(x=630, y=20)
    Button(ventanaD, text="Corregistrar imagenes",command=corregister, bg="#000000", fg="white").place(x=800, y=20)
    plt.subplot(2, 2, 1)  # axial
    #plt.imshow(corteA, cmap=plt.cm.get_cmap('bone'), aspect=pixel_len_mm[1] / pixel_len_mm[2])
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)

    plt.subplot(2, 2, 2)  # axial
    # plt.imshow(corteA, cmap=plt.cm.get_cmap('bone'), aspect=pixel_len_mm[1] / pixel_len_mm[2])
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)

    plt.subplot(2, 2, 3)  # axial
    # plt.imshow(corteA, cmap=plt.cm.get_cmap('bone'), aspect=pixel_len_mm[1] / pixel_len_mm[2])
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)



    frameD = Frame(ventanaD)

    canvasD = FigureCanvasTkAgg(figD, master=ventanaD)
    canvasD.draw()
    frameD.pack(side=TOP, fill=X)
    canvasD.get_tk_widget().pack(side=TOP, expand=1)

def loadImage(val):
    reader = itk.ImageSeriesReader()
    global imageDICOM1
    global imageDICOM2
    global image1
    global image2
    global imageITK1
    global imageITK2
    global valCorte
    directory = "imagenes_dicom/"
    filename = filedialog.askopenfilename(initialdir=directory, title="Select file")
    if val == 1:
        reader.SetFileNames(filename)
        imageITK1 = itk.ReadImage(filename, itk.sitkFloat32)
        imageITK1 = itk.Extract(imageITK1, (imageITK1.GetWidth(), imageITK1.GetHeight(), 0), (0, 0, 0))
        print(imageITK1)
        nameF = filename.rsplit('/', 1)
        nameF = nameF[1].rsplit('.', 1)
        valCorte = int(nameF[0])
        imageDICOM1 = pydicom.dcmread(filename)
        if len(imageDICOM1.pixel_array.shape) > 2:
            image1 = imageDICOM1.pixel_array[0]
        else:
            image1 = imageDICOM1.pixel_array
        plt.subplot(2, 2, 1)  # axial
        plt.imshow(image1, cmap=plt.cm.get_cmap('bone'))#aspect=pixel_len_mm[1] / pixel_len_mm[2]
        plt.gca().set_axis_off()
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=1)
        canvasD.figure = figD
        canvasD.draw()
        frameD.pack(side=TOP, fill=X)
        canvasD.get_tk_widget().pack(side=TOP, expand=1)

    else:
        reader.SetFileNames(filename)
        #imageITK2 = reader.Execute()
        imageITK2 = itk.ReadImage(filename,itk.sitkFloat32)
        imageITK2 = itk.Extract(imageITK2, (imageITK2.GetWidth(), imageITK2.GetHeight(), 0), (0, 0, valCorte))

        #print(imageITK2)
        imageDICOM2 = pydicom.dcmread(filename)
        if len(imageDICOM2.pixel_array.shape) > 2:
            image2 = imageDICOM2.pixel_array[valCorte]
        else:
            image2 = imageDICOM2.pixel_array

        plt.subplot(2, 2, 2)  # axial
        plt.imshow(image2, cmap=plt.cm.get_cmap('bone'))  # aspect=pixel_len_mm[1] / pixel_len_mm[2]
        plt.gca().set_axis_off()
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=1)

        canvasD.figure = figD
        canvasD.draw()
        frameD.pack(side=TOP, fill=X)
        canvasD.get_tk_widget().pack(side=TOP, expand=1)
        # itk.Show(cimg, "ImageRegistration2 Composition")
        canvasD.figure = figD
        canvasD.draw()
        frameD.pack(side=TOP, fill=X)
        canvasD.get_tk_widget().pack(side=TOP, expand=1)



    #res = pyelastix.register(imageDICOM2.pixel_array,imageDICOM2.pixel_array,pyelastix.get_default_params())
        #cv2.imshow("REGISTERED IMAGE",res)


def corregister():
    global img
    global imageITK1
    global imageITK2
    copy_img1 = np.asarray(image1, dtype=np.uint8)
    copy_img2 = np.asarray(image2, dtype=np.uint8)

    '''
    
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    warp_mode = cv2.MOTION_TRANSLATION
    number_of_iterations = 5000;
    termination_eps = 1e-10;
    #print(type(copy_img1[0][0]))
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    (cc, warp_matrix) = cv2.findTransformECC(copy_img1, copy_img2,
                                             warp_matrix, warp_mode, criteria, None, 1)
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        im2_aligned = cv2.warpPerspective(copy_img2, warp_matrix, (copy_img1.shape[1], copy_img1.shape[0]),
                                          flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        im2_aligned = cv2.warpAffine(copy_img2, warp_matrix, (copy_img1.shape[1], copy_img1.shape[0]),
                                     flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);
    cv2.imshow("Aligned Image 2", im2_aligned)
    cv2.waitKey(0)
    '''

    #Using SimpleITK

    registration_method = itk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(itk.sitkLinear)
    # Setup for the multi-resolution framework.
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(learningRate=0.006, numberOfIterations = 10000,
                                                      convergenceMinimumValue = 1e-6, convergenceWindowSize = 10)
    # registration_method.SetOptimizerScalesFromPhysicalShift()

    #print(type(imageITK1.shape))

    print(imageITK2.GetDimension())
    initial_transform = itk.CenteredTransformInitializer(imageITK2,
                                                          imageITK1,
                                                          itk.Euler2DTransform(),
                                                          itk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    # Need to compose the transformations after registration.
    outTx = registration_method.Execute(imageITK2, imageITK1)

    if (not "SITK_NOSHOW" in os.environ):
        resampler = itk.ResampleImageFilter()
        resampler.SetReferenceImage(imageITK2);
        resampler.SetInterpolator(itk.sitkLinear)
        resampler.SetDefaultPixelValue(1)
        resampler.SetTransform(outTx)

        out = resampler.Execute(imageITK1)

        simg1 = itk.Cast(itk.RescaleIntensity(imageITK2), itk.sitkUInt8)
        simg2 = itk.Cast(itk.RescaleIntensity(out), itk.sitkUInt8)
        cimg = itk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)
        print(cimg)
        imgF = itk.GetArrayFromImage(cimg)
        plt.subplot(2, 2, 3)  # axial
        plt.imshow(imgF, cmap=plt.cm.get_cmap('bone'))  # aspect=pixel_len_mm[1] / pixel_len_mm[2]
        plt.gca().set_axis_off()
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=1)
        canvasD.figure = figD
        canvasD.draw()
        frameD.pack(side=TOP, fill=X)
        canvasD.get_tk_widget().pack(side=TOP, expand=1)
        #itk.Show(cimg, "ImageRegistration2 Composition")
