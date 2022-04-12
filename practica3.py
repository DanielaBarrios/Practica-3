import numpy as np
from matplotlib import pyplot as plt
import math as ma
import cv2 #opencv

def histograma(ma1,ma2,mar,mae1,mae2,maer):
    ma1= cv2.resize(ma1, (500, 500))
    ma2= cv2.resize(ma2, (500, 500))
    mar= cv2.resize(mar, (500, 500))    
    #hitograma imagen 1 sin ecualizar
    plt.subplot(3, 2, 1)
    plt.title("histograma imagen1")
    color = ('b','g','r')
    for i, c in enumerate(color):
        hist = cv2.calcHist([ma1], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
    #histograma imagen 2 sin ecualizar
    plt.subplot(3, 2, 3)
    plt.title("histograma imagen2")
    color = ('b','g','r')
    for i, c in enumerate(color):
        hist = cv2.calcHist([ma2], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
    #histograma imagen resultado sin ecualizar
    plt.subplot(3, 2, 5)
    plt.title("histograma imagen resultado")
    color = ('b','g','r')
    for i, c in enumerate(color):
        hist = cv2.calcHist([mar], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
    #histograma imagen 1 ecualizada
    plt.subplot(3, 2, 2)
    plt.title("histograma img1 ecualizado imagen3")
    color = ('b','g','r')
    for i, c in enumerate(color):
        hist = cv2.calcHist([mae1], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
    #histograma imagen 2 cualizada
    plt.subplot(3, 2, 4)
    plt.title("histograma img2 ecualizado imagen3")
    color = ('b','g','r')
    for i, c in enumerate(color):
        hist = cv2.calcHist([mae2], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])
    #histograma imagen resultado ecualizada
    plt.subplot(3, 2, 6)
    plt.title("histograma resultado ecualizado imagen3")
    color = ('b','g','r')
    for i, c in enumerate(color):
        hist = cv2.calcHist([maer], [i], None, [256], [0, 256])
        plt.plot(hist, color = c)
        plt.xlim([0,256])        
        
    plt.show()
    cv2.waitKey(0)

def ecualizado(me1,me2,mer):
    #ecualizacion imagen 1
    img_to_yuv = cv2.cvtColor(me1,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    meq1 = cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
    cv2.imshow('Imagen ecualizada 1', meq1)
    #ecualizacion imagen 2
    img_to_yuv = cv2.cvtColor(me2,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    meq2= cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
    cv2.imshow('Imagen ecualizada 2', meq2)
    #ecualizacion imagen 3  
    img_to_yuv = cv2.cvtColor(mer,cv2.COLOR_BGR2YUV)
    img_to_yuv[:,:,0] = cv2.equalizeHist(img_to_yuv[:,:,0])
    meqr= cv2.cvtColor(img_to_yuv, cv2.COLOR_YUV2BGR)
    cv2.imshow('Resultado ecualizado', meqr)
    histograma(me1,me2,mer,meq1,meq2,meqr)
    
        
    
def suma(m1, m2):
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    mr = m1 + m2
    cv2.imshow("suma",mr)
    ecualizado(m1,m2,mr)
    cv2.waitKey(0)

def suma2(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)    
    mr = cv2.add( ima1,ima2)
    cv2.imshow("suma2",mr)
    ecualizado(m1,m2,mr)
    cv2.waitKey(0)
    
def suma3(m1, m2):
    cv2.destroyAllWindows()  
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2) 
    mr = np.add( ima1,ima2)
    cv2.imshow("suma3",mr)
    ecualizado(m1,m2,mr)
    cv2.waitKey(0)
    
def resta1(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = ima1-ima2
    cv2.imshow("resta1",mr)
    ecualizado(m1,m2,mr)    
    cv2.waitKey(0)
    
def resta2(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = np.subtract(ima1,ima2)
    cv2.imshow("resta2",mr)
    ecualizado(m1,m2,mr)    
    cv2.waitKey(0)
    
def resta3(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = cv2.subtract( ima1,ima2)
    cv2.imshow("resta3",mr)
    ecualizado(m1,m2,mr)    
    cv2.waitKey(0)
    
def division1(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = ima1/ima2
    cv2.imshow("division1",mr)
    ecualizado(m1,m2,mr)    
    cv2.waitKey(0)
    
def division2(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = np.divide(ima1,ima2)
    cv2.imshow("division2",mr)
    ecualizado(m1,m2,mr)    
    cv2.waitKey(0)
    
def division3(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = cv2.divide( ima1,ima2)
    cv2.imshow("division3",mr)
    ecualizado(m1,m2,mr)    
    cv2.waitKey(0)
    
def multiplicacion1(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = ima1*ima2
    cv2.imshow("multiplicacion1",mr)
    ecualizado(m1,m2,mr)    
    cv2.waitKey(0)
    
def multiplicacion2(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = np.multiply(ima1,ima2)
    cv2.imshow("multiplicacion2",mr)
    ecualizado(m1,m2,mr)    
    cv2.waitKey(0)
    
def multiplicacion3(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = cv2.multiply ( ima1,ima2)
    cv2.imshow("multiplicacion3",mr)
    ecualizado(m1,m2,mr)    
    cv2.waitKey(0)

def logn1(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = np.log(ima1)
    #cv2.imshow("logaritmo natural 1",mr)
    ecualizado(m1,m2,mr)
    cv2.waitKey(0)   
    
def raiz1(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = ima1**(0.5)
    cv2.imshow("raiz1",mr)
    ecualizado(m1,m2,mr)
    cv2.waitKey(0) 
    
def raiz2(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = pow(ima1,0.5)
    cv2.imshow("raiz2",mr)
    ecualizado(m1,m2,mr)
    cv2.waitKey(0)    
    
def raiz3(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = np.sqrt(ima1)
    cv2.imshow("raiz3",mr)
    ecualizado(m1,m2,mr)
    cv2.waitKey(0) 
    
def derivada1(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    dx = np.diff(ima1)
    dy = np.diff(ima2)
    d = dy/dx
    cv2.imshow("derivada1",d)
    cv2.waitKey(0)    
    
def potencia1(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = m1**2
    cv2.imshow("potencia1",mr)
    ecualizado(m1,m2,mr)
    cv2.waitKey(0)    
    
def potencia2(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = np.power(m1,2)
    cv2.imshow("potencia2",mr)
    ecualizado(m1,m2,mr)
    cv2.waitKey(0)    
    
def potencia3(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = cv2.pow(m1,2)
    cv2.imshow("potencia3",mr)
    ecualizado(m1,m2,mr)
    cv2.waitKey(0)  
    
def conjuncion(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = cv2.bitwise_and(m1, m2)
    cv2.imshow("conjuncion",mr)
    ecualizado(m1,m2,mr)
    cv2.waitKey(0)

def disyuncion(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = cv2.bitwise_or(m1,m2)
    cv2.imshow("disyuncion",mr)
    ecualizado(m1,m2,mr)
    cv2.waitKey(0)

def negacion(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = cv2.bitwise_not(m1)
    cv2.imshow("negacion",mr)
    ecualizado(m1,m2,mr)
    cv2.waitKey(0)

def trasafin(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    ancho = m1.shape[1] #columnas
    alto = m1.shape[0] # filas
    mr = np.float32([[10,0,200],[0,5,100]])
    imageOut = cv2.warpAffine(m1,mr,(ancho,alto))
    cv2.imshow("trasalacion afin",imageOut)
    ecualizado(m1,m2,imageOut)
    cv2.waitKey(0)    

def escalado(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = cv2.resize(m1,(600,300), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("escalado",mr)
    ecualizado(m1,m2,mr)
    cv2.waitKey(0)

def rotacion(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,250))
    m2 = cv2.resize(m2, (300,250))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    ancho = m1.shape[1] #columnas
    alto = m1.shape[0] # filas
    mr = cv2.getRotationMatrix2D((ancho//2,alto//2),15,1)
    imageOut = cv2.warpAffine(m1,mr,(ancho,alto))
    cv2.imshow("rotacion",imageOut)
    ecualizado(m1,m2,imageOut)
    cv2.waitKey(0)    

def traspuesta(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,252))
    m2 = cv2.resize(m2, (302,252))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    mr = np.transpose(m2).shape
    #mr = cv2.resize(mr,(520,520))
    cv2.imshow("traspuesta",mr)
    ecualizado(m1,m2,mr)
    cv2.waitKey(0)

def proyeccion(m1, m2):
    cv2.destroyAllWindows()
    m1 = cv2.resize(m1, (300,252))
    m2 = cv2.resize(m2, (302,252))
    cv2.imshow('Imagen1', m1)
    cv2.imshow('Imagen2', m2)
    m1 = cv2.cvtColor(m1,cv2.COLOR_BGR2GRAY)
    ret,thresh1=cv2.threshold(m1,130,255,cv2.THRESH_BINARY)
    (h, w) = thresh1.shape
    a = [0 for z in range(0, h)]
    for j in range (0, h):
        for i in range (0, h):
            if thresh1 [j, i] == 0:
                a[j]+=1
                thresh1 [j, i] = 255
    for j in range (0, h):
        for i in range ((h-a [j]), h):
            thresh1 [i, j] = 0
    plt.imshow(thresh1,cmap=plt.gray())
    plt.show()
    cv2.imshow('img',thresh1)

ima1 = cv2.imread('foto1.png')
ima2 = cv2.imread('foto2.png')
ima1 = cv2.resize(ima1, (300,250))
ima2 = cv2.resize(ima2, (300,250))

cv2.imshow('Imagen1', ima1)
cv2.imshow('Imagen2', ima2)
x=cv2.waitKey(0)

while True:



    if x == ord("d"):
        suma(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        suma2(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        suma3(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        resta1(ima1,ima2)
        x=cv2.waitKey(0)

        
    if x == ord("d"):
        resta2(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        resta3(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        division3(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        multiplicacion1(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        multiplicacion2(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        multiplicacion3(ima1,ima2)
        x=cv2.waitKey(0)             
        
    if x == ord("d"):
        potencia1(ima1,ima2)
        x=cv2.waitKey(0)        
                      
    if x == ord("d"):
        potencia2(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        potencia3(ima1,ima2)
        x=cv2.waitKey(0) 
        
    if x == ord("d"):
        conjuncion(ima1,ima2)
        x=cv2.waitKey(0)  
        
    if x == ord("d"):
        disyuncion(ima1,ima2)
        x=cv2.waitKey(0)        
        
    if x == ord("d"):
        negacion(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        trasafin(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        escalado(ima1,ima2)
        x=cv2.waitKey(0)
        
    if x == ord("d"):
        rotacion(ima1,ima2)
        x=cv2.waitKey(0)    
        
    if x == ord("d"):
        traspuesta(ima1,ima2)
        x=cv2.waitKey(0)      
        
    if x == ord("d"):
        proyeccion(ima1,ima2)
        x=cv2.waitKey(0)
        
        cv2.destroyAllWindows()        
    break
    
