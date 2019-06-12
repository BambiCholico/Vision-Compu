# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from math import floor
from skimage import io

#1) IMAGEN LENA
print("1) Imagen original de Lena")
img = io.imread('Lena-grayscale.jpg')
img = rgb2gray(img)
io.imshow(img)
plt.axis('off')
plt.show()
print(img.shape)

from skimage.transform import resize
def resize_img(img):
  return resize(img, (200, 200))
img = resize_img(img)
print("Lena escalada")
io.imshow(img)
plt.axis('off')
plt.show()
print(img.shape)
print(img.min())
print(img.max())

#2) INTENSIDAD DE 80% true o false
print("2) Matriz de intensidades")
inten = img.max()*0.8
#per = np.percentile(img, 80)
#print(inten)
R = img >= inten
print(R)
print(R.shape)

#3) INTENSIDAD menores 80% son cero
print("3) Cambio de valores dependiendo a su intensidad")
BN = img.copy()
BN[img < inten] = 0
#print(BN)
plt.imshow(BN, cmap = plt.cm.gray)
plt.axis('off')
plt.show()
print(BN.shape)

print("4) A la operación hecha en los puntos 2 y 3 se le llama ecualizar la iluminacion y el contraste de la imagen")


#5)CONVOLUCION FUNCION

def conv2d(image, kernel):
    m, n = kernel.shape
    if (m != n):
        maxi = max(m,n)
        cuad = (maxi,maxi)
        ceros = np.zeros(cuad)
        ceros[:m,:n] = kernel[:m,:n]
        kernel = ceros
        m = maxi
        n = m
        
    if (m == n):
        kernel = np.flip(kernel)
        y, x = image.shape
        h = floor(m/2)
        t = y - h
        s = x - h
        
        new_image = np.zeros((y,x))
        for i in range(h,t):
            for j in range(h,s):
                suma = 0
                for a in range(m):
                    for b in range(m):
                        suma = suma + kernel[a][b] * image[i-h+a][j-h+b]
                new_image[i][j] = suma
        new_image = abs(new_image)
        return new_image

#6) CONVOLUCION CALCULO
print("5) 6) y 7) Convolución")
k=np.array([[-1, -1, -1],[2, 2, 2],[-1, -1, -1]])
conv = conv2d(img, k)

#7) MOSTRAR RESULTADO DE 6
print("Convolucion con kernel 3x3")
#print(conv)
plt.imshow(conv, cmap = plt.cm.gray)
plt.axis('off')
plt.show()
print(conv.shape)

#8) REEMPLAZAR K POR SU TRANSPUESTA

tK = k.transpose()
convT = conv2d(img, tK)

print("8) Cambió el resultado, se notan más los contornos verticales y antes se notaban más los horizontales")

#9) MOSTRAR RESULTADO DE 8
print("9) Convolucion con kernel 3x3 transpuesto")
#print(convT)
plt.imshow(convT,cmap=plt.cm.gray)
plt.axis('off')
plt.show()
print(convT.shape)

#10) USAR UN NUEVO KERNEL 1 Y 4 VECES CONSECUTIVAMENTE
kNew=np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]])*(1/256)
convKNew1 = conv2d(img, kNew)

print("10) Convolucion con kernel 5x5")
#print(convKNew1)
plt.imshow(convKNew1,cmap=plt.cm.gray)
plt.axis('off')
plt.show()
print(convKNew1.shape)

print("10) Convolucion con kernel 5x5 4 veces")
convKNew2 = conv2d(convKNew1, kNew)
convKNew3 = conv2d(convKNew2, kNew)
convKNew4 = conv2d(convKNew3, kNew)
#print(convKNew4)
plt.imshow(convKNew4,cmap=plt.cm.gray)
plt.axis('off')
plt.show()
print(convKNew4.shape)

#11)

def cuantasConv(image, kernel):
    convol = conv2d(image, kernel)
    cuenta = 1
    while np.std(convol) > 0.1:
        convol = conv2d(convol, kernel)
        cuenta += 1
    return cuenta

print("11) Cuantas veces se debe de aplicar un kernel para que la variacion promedio sea menor a 0.1")

#n = cuantasConv(img, kNew)
#print(n)
print("3283 veces se debe de aplicar el kernel gaussiano sobre Lena antes que la variación promedio de los pixeles sea menos a 0.1")


#12) APLICA KERNEL DEL PUNTO 6 SOBRE LA RESULTANDE DE 10 CON 4 
print("12) Convolucion imagen resultante del punto 10 con de kernel del punto 6")
conv12 = conv2d(convKNew4, k)
#print(conv12)
plt.imshow(conv12,cmap=plt.cm.gray)
plt.axis('off')
plt.show()
print(conv12.shape)

#13)
print("13) El resultado del punto 12 tiene un mayor contraste comparado con el del punto 6, aunque parece ser que describen la silueta de la misma forma, acentuando las lineas horizontales")

#14)
print("14) Un filtro gaussiano elimina el ruido gaussiano y se utiliza para suavizar controladamente la imagen; mientras que un filtro pasa altas permite elevar el contraste de la imagen")

#15)
print("15) Convolucion 4 veces sobre Lena con kernel del punto 9")
conv15_1 = conv2d(img, tK)
conv15_2 = conv2d(conv15_1, tK)
conv15_3 = conv2d(conv15_2, tK)
conv15_4 = conv2d(conv15_3, tK)

#print(conv15_4)
"""plt.imshow(conv15_4,cmap=plt.cm.gray)
plt.axis('off')
plt.show()
print(conv15_4.shape)"""

print("Resta de la imagen resultante de la convolucion anterior menos Lena")
conv15 = conv15_4 - img
plt.imshow(conv15,cmap=plt.cm.gray)
plt.axis('off')
plt.show()
print(conv15.shape)

#16) 
print("16) Realmente la resta no hizo ninguna diferencia ni agregó algo a las 4 convoluciones y, en general, solo se acentuaron los bordes verticales de tal forma que parecen marcas")



