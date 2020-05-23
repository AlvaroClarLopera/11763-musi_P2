import numpy as np
import cv2
import natsort
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
    global valCorteA
    global valCorteB
    global valCorteC
    valCorteA = 0
    valCorteB = 0
    valCorteC = 0
    ventanaD = Tk()
    ventanaD.resizable(0, 0)
    ventanaD.title("Herramienta de corregistro")
    ventanaD.geometry("1280x720")
    ventanaD.configure(background=colorFondo)

    figD = plt.figure(figsize=(1200/my_dpi, 620/my_dpi), dpi=my_dpi, facecolor=colorFondo)
    #pixel_len_mm = [imageDICOM.SliceThickness, imageDICOM.PixelSpacing[0], imageDICOM.PixelSpacing[1]] #slice thickness, pixel spacing 0 1

    Button(ventanaD, text="Cargar imagen 1", command=lambda: loadImage(1), bg="#000000", fg="white").place(x=470, y=20)
    Button(ventanaD, text="Cargar imagen 2", command=lambda: loadImage(2), bg="#000000", fg="white").place(x=630, y=20)
    Button(ventanaD, text="Corregistrar imagenes",command=corregister, bg="#000000", fg="white").place(x=800, y=20)


    plt.subplot(2, 3, 1)  # axial
    #plt.imshow(corteA, cmap=plt.cm.get_cmap('bone'), aspect=pixel_len_mm[1] / pixel_len_mm[2])
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)

    plt.subplot(2, 3, 2)  # axial
    # plt.imshow(corteA, cmap=plt.cm.get_cmap('bone'), aspect=pixel_len_mm[1] / pixel_len_mm[2])
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)

    plt.subplot(2, 3, 3)  # axial
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

    lArrow = PhotoImage(file="igu_data/left_arrow.png", master=ventanaD)
    rArrow = PhotoImage(file="igu_data/right_arrow.png", master=ventanaD)
    Button(ventanaD, width=30, height=30, image=lArrow, command=lambda: cambiarCorte(-1,0)).place(
        x=30, y=0)
    Button(ventanaD, width=30, height=30, image=rArrow, command=lambda: cambiarCorte(1,0)).place(
        x=60, y=0)
    Button(ventanaD, width=30, height=30, image=lArrow, command=lambda: cambiarCorte(-1,1)).place(
        x=100, y=0)
    Button(ventanaD, width=30, height=30, image=rArrow, command=lambda: cambiarCorte(1,1)).place(
        x=130, y=0)
    Button(ventanaD, width=30, height=30, image=lArrow, command=lambda: cambiarCorte(-1,2)).place(
        x=170, y=0)
    Button(ventanaD, width=30, height=30, image=rArrow, command=lambda: cambiarCorte(1,2)).place(
        x=200, y=0)
    mainloop()


def loadImage(val):
    reader = itk.ImageSeriesReader()
    global imageDICOM1
    global imageDICOM2
    global imageDICOM3
    global image1
    global image2
    global image3
    global imageITK1
    global imageITK2
    global imageITK_atlas
    global valCorteA
    global valCorteB
    global valCorteC
    global ventanaD
    global filenameA
    global filenameB
    global filenameC
    global directory


    directory = "P2 - DICOM/"
    filename = filedialog.askopenfilename(initialdir=directory, title="Select file")
    if val == 1:
        filenameA = cp.deepcopy(filename)
        reader.SetFileNames(filename)
        print(filename)
        nameF = filename.rsplit('/', 1)
        nameF = nameF[1].rsplit('.', 1)

        valCorteA = int(nameF[0])
        imageITK1 = itk.ReadImage(filenameA, itk.sitkFloat32)
        imageITK1 = itk.Extract(imageITK1, (imageITK1.GetWidth(), imageITK1.GetHeight(), 0), (0, 0, 0))

        imageDICOM1 = pydicom.dcmread(filenameA)
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
        canvasD.get_tk_widget().pack(side=TOP, expand=0.5)



    else:
        filenameB = cp.deepcopy(filename)
        reader.SetFileNames(filenameB)
        #imageITK2 = reader.Execute()
        valCorteB = cp.deepcopy(valCorteA)
        print(valCorteB)
        imageITK2 = itk.ReadImage(filenameB,itk.sitkFloat32)
        imageITK2 = itk.Extract(imageITK2, (imageITK2.GetWidth(), imageITK2.GetHeight(), 0), (0, 0, valCorteB))

        #print(imageITK2)
        filenameC = "P2 - DICOM/AAL3_1mm.dcm"
        imageDICOM2 = pydicom.dcmread(filenameB)
        imageDICOM3 = pydicom.dcmread(filenameC)
        imageITK_atlas = itk.ReadImage(filenameC, itk.sitkFloat32)
        valCorteC = cp.deepcopy(valCorteA)
        imageITK_atlas = itk.Extract(imageITK_atlas, (imageITK_atlas.GetWidth(), imageITK_atlas.GetHeight(), 0), (0, 0, valCorteC))
        image3 = imageDICOM3.pixel_array[valCorteC]
        if len(imageDICOM2.pixel_array.shape) > 2:
            print(valCorteB)
            image2 = imageDICOM2.pixel_array[valCorteB]
        else:
            image2 = imageDICOM2.pixel_array

        plt.subplot(2, 3, 2)  # axial
        plt.imshow(image2, cmap=plt.cm.get_cmap('bone'))  # aspect=pixel_len_mm[1] / pixel_len_mm[2]
        plt.gca().set_axis_off()
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=1)

        plt.subplot(2, 3, 3)  # axial
        plt.imshow(image3, cmap=plt.cm.get_cmap('bone'))  # aspect=pixel_len_mm[1] / pixel_len_mm[2]
        plt.gca().set_axis_off()
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=1)

        canvasD.figure = figD
        canvasD.draw()
        frameD.pack(side=TOP, fill=X)
        canvasD.get_tk_widget().pack(side=TOP, expand=1)
        # itk.Show(cimg, "ImageRegistration2 Composition")

    #res = pyelastix.register(imageDICOM2.pixel_array,imageDICOM2.pixel_array,pyelastix.get_default_params())
        #cv2.imshow("REGISTERED IMAGE",res)

def cambiarCorte(valor,corte):
    global valCorteA
    global valCorteB
    global valCorteC
    global imageITK1
    global imageITK2
    global imageITK_atlas
    global image1
    global image2
    global image3
    global filenameA
    global filenameB
    global filenameC
    global directory
    # print(imageDICOM.pixel_array.shape)
    if corte == 0:
        if "imagenes_dicom/0-27993/" or "CT_Lung/" or "RM_Brain_3D-SPGR/" in filenameA:
            if valor == -1:  # anterior corte
                valCorteA -= 1
                if valCorteA < 0:
                    valCorteA = 211
            else:  # siguiente corte
                valCorteA += 1
                if valCorteA > 211:
                    valCorteA = 0
            cont = 0
            print(filenameA)
            if "imagenes_dicom/0-27993/" in filenameA:
                directory = "imagenes_dicom/0-27993/"
            if "CT_Lung/" in filenameA:
                directory = "P2 - DICOM/CT_Lung/"
            if "RM_Brain_3D-SPGR/" in filenameA:
                directory = "P2 - DICOM/RM_Brain_3D-SPGR/"
            list_dir = natsort.natsorted(os.listdir(directory))
            print(list_dir)
            for fname in list_dir:
                #print(directory + fname)
                imageDICOMa = pydicom.dcmread(directory + fname)
                print(fname)
                if cont == valCorteA:
                    break
                cont += 1
            image1 = imageDICOMa.pixel_array
            imageITK1 = itk.ReadImage(directory+fname, itk.sitkFloat32)
            imageITK1 = itk.Extract(imageITK1, (imageITK1.GetWidth(), imageITK1.GetHeight(), 0), (0, 0, 0))
            firstView = True
        else:
            if valor == -1:  # anterior corte
                valCorteA -= 1
                if valCorteA < 0:
                    valCorteA = imageDICOM1.pixel_array.shape[0] - 1
            else:  # siguiente corte
                valCorteA += 1
                if valCorteA > imageDICOM1.pixel_array.shape[0] - 1:
                    valCorteA = 0
            image1 = imageDICOM1.pixel_array[valCorteA]
            imageITK1 = itk.ReadImage(filenameA, itk.sitkFloat32)
            imageITK1 = itk.Extract(imageITK1, (imageITK1.GetWidth(), imageITK1.GetHeight(), 0), (0, 0, valCorteA))

    elif corte == 1:
        if valor == -1:  # anterior corte
            valCorteB -= 1
            if valCorteB < 0:
                valCorteB = imageDICOM2.pixel_array.shape[0] - 1
        else:  # siguiente corte
            valCorteB += 1
            if valCorteB > imageDICOM2.pixel_array.shape[0] - 1:
                valCorteB = 0
        image2 = imageDICOM2.pixel_array[valCorteB]
        imageITK2 = itk.ReadImage(filenameB, itk.sitkFloat32)
        imageITK2 = itk.Extract(imageITK2, (imageITK2.GetWidth(), imageITK2.GetHeight(), 0), (0, 0, valCorteB))
    else:
        if valor == -1:  # anterior corte
            valCorteC -= 1
            if valCorteC < 0:
                valCorteC = imageDICOM3.pixel_array.shape[0] - 1
        else:  # siguiente corte
            valCorteC += 1
            if valCorteC > imageDICOM3.pixel_array.shape[0] - 1:
                valCorteC = 0
        image3 = imageDICOM3.pixel_array[valCorteC]
        imageITK_atlas = itk.ReadImage(filenameC, itk.sitkFloat32)
        imageITK_atlas = itk.Extract(imageITK_atlas, (imageITK_atlas.GetWidth(), imageITK_atlas.GetHeight(), 0),(0, 0, valCorteC))

    plt.subplot(2, 3, 1)  # axial
    plt.imshow(image1, cmap=plt.cm.get_cmap('bone'))
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=0.5)
    plt.subplot(2, 3, 2)  # coronal
    plt.imshow(image2, cmap=plt.cm.get_cmap('bone'))
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)
    plt.subplot(2, 3, 3)  # sagital
    plt.imshow(image3, cmap=plt.cm.get_cmap('bone'))
    plt.gca().set_axis_off()
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                        hspace=0, wspace=1)

    canvasD.figure = figD
    canvasD.draw()
    frameD.pack(side=TOP, fill=X)
    canvasD.get_tk_widget().pack(side=TOP, expand=1)


def corregister():
    global img
    global imageITK1
    global imageITK2
    global image3
    global imageITK_atlas
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

    #print(imageITK2.GetDimension())
    initial_transform = itk.CenteredTransformInitializer(imageITK2,
                                                          imageITK1,
                                                          itk.Euler2DTransform(),
                                                          itk.CenteredTransformInitializerFilter.GEOMETRY)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    # Need to compose the transformations after registration.
    outTx = registration_method.Execute(imageITK2, imageITK1)

    #Función inversa
    r_registration_method = itk.ImageRegistrationMethod()

    # Similarity metric settings.
    r_registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    r_registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    r_registration_method.SetMetricSamplingPercentage(0.01)

    r_registration_method.SetInterpolator(itk.sitkLinear)
    # Setup for the multi-resolution framework.
    r_registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    # Optimizer settings.
    r_registration_method.SetOptimizerAsGradientDescent(learningRate=0.006, numberOfIterations=10000,
                                                      convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    # registration_method.SetOptimizerScalesFromPhysicalShift()

    # print(type(imageITK1.shape))

    #print(imageITK2.GetDimension())
    r_initial_transform = itk.CenteredTransformInitializer(imageITK1,
                                                         imageITK_atlas,
                                                         itk.Euler2DTransform(),
                                                         itk.CenteredTransformInitializerFilter.GEOMETRY)
    r_registration_method.SetInitialTransform(r_initial_transform, inPlace=False)
    # Need to compose the transformations after registration.
    r_outTx = r_registration_method.Execute(imageITK1, imageITK_atlas)


    if (not "SITK_NOSHOW" in os.environ):
        resampler = itk.ResampleImageFilter()
        resampler.SetReferenceImage(imageITK2);
        resampler.SetInterpolator(itk.sitkLinear)
        resampler.SetDefaultPixelValue(1)
        resampler.SetTransform(outTx)

        out = resampler.Execute(imageITK1)

        resampler.SetReferenceImage(imageITK1);
        resampler.SetInterpolator(itk.sitkLinear)
        resampler.SetDefaultPixelValue(1)
        resampler.SetTransform(r_outTx)

        r_out = resampler.Execute(imageITK_atlas)

        simg1 = itk.Cast(itk.RescaleIntensity(imageITK2), itk.sitkUInt8)
        simg2 = itk.Cast(itk.RescaleIntensity(out), itk.sitkUInt8)
        cimg = itk.Compose(simg1, simg2, simg1 // 2. + simg2 // 2.)


        #print(cimg)

        r_simg1 = itk.Cast(itk.RescaleIntensity(imageITK1), itk.sitkUInt8)
        r_simg2 = itk.Cast(r_out, itk.sitkUInt8)

        r_imgIn = itk.GetArrayFromImage(r_simg2)
        print(np.unique(r_imgIn))
        hip_inf = 121
        hip_sup = 150
        if (r_imgIn > hip_inf).any() and (r_imgIn < hip_sup).any():
            #print("holaa")
            r_imgIn[r_imgIn < 121] = 0
            r_imgIn[r_imgIn > 150] = 0
            #print(np.unique(r_imgIn))
        r_simg2_s = itk.GetImageFromArray(r_imgIn)
        r_simg2_s.CopyInformation(r_out)
        aux = itk.GetArrayFromImage(r_simg2_s)
        print(np.unique(aux))
        cv2.imshow("hip",aux)
        cv2.waitKey(0)
        r_cimg = itk.Compose(r_simg1, r_simg2_s, r_simg1 // 2. + r_simg2_s // 2.)

        imgF = itk.GetArrayFromImage(cimg)
        imgIn = itk.GetArrayFromImage(simg2)

        r_imgF = itk.GetArrayFromImage(r_cimg)



        plt.subplot(2, 3, 3)  # axial
        plt.imshow(image3, cmap=plt.cm.get_cmap('bone'))  # aspect=pixel_len_mm[1] / pixel_len_mm[2]
        plt.gca().set_axis_off()
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=1)

        plt.subplot(2, 3, 4)  # axial
        plt.imshow(imgF, cmap=plt.cm.get_cmap('bone'))  # aspect=pixel_len_mm[1] / pixel_len_mm[2]
        plt.gca().set_axis_off()
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=1)
        plt.subplot(2, 3, 5)  # axial
        plt.imshow(imgIn, cmap=plt.cm.get_cmap('bone'))  # aspect=pixel_len_mm[1] / pixel_len_mm[2]
        plt.gca().set_axis_off()
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=1)

        plt.subplot(2, 3, 6)  # axial
        plt.imshow(r_imgF, cmap=plt.cm.get_cmap('bone'))  # aspect=pixel_len_mm[1] / pixel_len_mm[2]
        plt.gca().set_axis_off()
        plt.margins(0, 0)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=1)

        canvasD.figure = figD
        canvasD.draw()
        frameD.pack(side=TOP, fill=X)
        canvasD.get_tk_widget().pack(side=TOP, expand=1)
        #itk.Show(cimg, "ImageRegistration2 Composition")
