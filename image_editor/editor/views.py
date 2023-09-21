from django.shortcuts import render,redirect
from django.shortcuts import HttpResponse,get_object_or_404
from .forms import *
from .models import images
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from innovative import settings
import os
from pathlib import Path
from PIL import Image, ImageFilter


def upload_image(request):
    if request.method == 'POST':
        form = imageForm(request.POST, request.FILES)
        if form.is_valid():  
            form.save()  
            # Getting the current instance object to display in the template  
            stud = images.objects.last()
            #print(stud.filters)
            #gam = gammaForm(request.POST)
            #print(stud)
            filters = request.POST['filters']
            if  filters == 'gamma':
                return redirect('/gamma')

            if filters == 'log':
                return redirect('/show_log')

            if filters == 'median':
                return redirect('/median')

            if filters == 'max':
                return redirect('/max')

            if filters == 'sketch':
                return redirect('/sketch')
                
            if filters == 'min':
                return redirect('/min')

            if filters == 'erode_dilate':
                return redirect('/erode_dilate')
                
            if filters == 'mode':
                return redirect('/mode')

            if filters == 'mean':
                return redirect('/mean')

            if filters == 'erode':
                return redirect('/erode')

            if filters == 'dilate':
                return redirect('/dilate')

            if filters == 'contrast':
                return redirect('/contrast')

            if filters == 'histogram':
                return redirect('/show_histogram_equalization')

            if filters == 'Laplacian':
                return redirect('/show_Laplacian')

            if filters == 'Ideal_low_Pass':
                return redirect('/ideal_lowpass')

            if filters == 'Ideal_high_Pass':
                return redirect('/ideal_highpass')

            if filters == 'Butterworth_low_Pass':
                return redirect('/butterworth_lowpass')

            if filters == 'Butterworth_high_Pass':
                return redirect('/butterworth_highpass')   
                
            return render(request, 'user.html', {'form': form, 'stu':stud})  
    
    else:
        form = imageForm()
    return render(request,'image_form.html',{'form': form})

def log_transform(image) :
    c = 255 / np.log(1 + np.max(image))
    log_image = c * (np.log(image + 1))
    log_image = np.array(log_image, dtype = np.uint8)
    return log_image

def show_log(request):
    if request.method == 'POST':
        return redirect('/upload')
    stud = images.objects.last()
    primary = stud.pk
    product = get_object_or_404(images, pk=primary)
    img_path = 'media/'+str(product.image)
    img = plt.imread(img_path)
    log_img = log_transform(img)
    pat = "media/media/abc.png"
    cv2.imwrite(pat, log_img)
    hist(img,1)
    hist(log_img, 0)
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Log transformation"})

def gamma_transform(img,gamma):
     print(type(gamma), gamma)
     gamma_corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
     print(gamma_corrected)
     return gamma_corrected

def gamma(request):
    if request.method == 'POST':
        gamf = gammaForm(request.POST,request.FILES)
        if gamf.is_valid():
            #gamf.save()
            stud = images.objects.last()
            primary = stud.pk
            product = get_object_or_404(images, pk=primary)
            img_path = 'media/'+str(product.image)
            img = plt.imread(img_path)
            temp = request.POST["gamma"]
            gam_img = gamma_transform(img,float(temp))
            pat = "media/media/abc.png"
            cv2.imwrite(pat, gam_img)
            hist(img,1)
            hist(gam_img, 0)
            return redirect('/show_gamma')
    else:
        gamf = gammaForm()
    return render(request,'gamma.html',{'gamf':gamf})

def show_gamma(request):
    stud = images.objects.last()
    if request.method == 'POST':
        return redirect('/upload')
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Gamma transformation"})

def median(request):
    if request.method == 'POST':
        return redirect('/')
    stud = images.objects.last()
    primary = stud.pk
    product = get_object_or_404(images, pk=primary)
    img_path = 'media/'+str(product.image)
    img = plt.imread(img_path)
    log_img = median_filter(img)
    pat = "media/media/abc.png"
    hist(img,1)
    hist(log_img, 0)
    cv2.imwrite(pat, log_img)
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Median filter"})

def median_filter(im1):
    im1 =   Image.fromarray(im1)
    im2 = im1.filter(ImageFilter.MedianFilter(size = 3)) 
    im2 = np.array(im2)
    return im2

def mode_filter(im1):
    im1 =   Image.fromarray(im1)
    im2 = im1.filter(ImageFilter.ModeFilter(size = 3)) 
    im2 = np.array(im2)
    return im2

def mode(request):
    if request.method == 'POST':
        return redirect('/upload')
    stud = images.objects.last()
    primary = stud.pk
    product = get_object_or_404(images, pk=primary)
    img_path = 'media/'+str(product.image)
    img = plt.imread(img_path)
    log_img = mode_filter(img)
    pat = "media/media/abc.png"
    cv2.imwrite(pat, log_img)
    hist(img,1)
    hist(log_img, 0)
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Mode filter"})

def show_contrast(request):
    stud = images.objects.last()
    if request.method == 'POST':
        return redirect('/upload')
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Contrast stretching"})

def contrast_stretching(request):
    if request.method == 'POST':
        con_form = contrastForm(request.POST,request.FILES)
        if con_form.is_valid():
            #gamf.save()
            stud = images.objects.last()
            primary = stud.pk
            product = get_object_or_404(images, pk=primary)
            img_path = 'media/'+str(product.image)
            img = plt.imread(img_path)
            r1 = request.POST["r1"]
            s1 = request.POST["s1"]
            r2 = request.POST["r2"]
            s2 = request.POST["s2"]
            pixelVal_vec = np.vectorize(pixelVal)
  
            contrast_stretched = pixelVal_vec(img, int(r1), int(s1), int(r2), int(s2))
    
            pat = "media/media/abc.png"
            hist(img,1)
            hist(contrast_stretched, 0)
            cv2.imwrite(pat, contrast_stretched)
            return redirect('/show_contrast')
    else:
        gamf = contrastForm()
    return render(request,'gamma.html',{'gamf':gamf})
  
def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1)*pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2

def histogram_equalization(img):
    equ = cv2.equalizeHist(img)
    # cv2.imwrite('res.png',res)
    return equ

def show_histogram_equalization(request):
    if request.method == 'POST':
        return redirect('/upload')
    stud = images.objects.last()
    primary = stud.pk
    product = get_object_or_404(images, pk=primary)
    img_path = 'media/'+str(product.image)
    img = plt.imread(img_path)
    new_image = histogram_equalization(img)
    pat = "media/media/abc.png"
    hist(img,1)
    hist(new_image, 0)
    cv2.imwrite(pat, new_image)
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Histogram equalization"})

def show_Laplacian(request):
    if request.method == 'POST':
        return redirect('/upload')
    stud = images.objects.last()
    primary = stud.pk
    product = get_object_or_404(images, pk=primary)
    img_path = 'media/'+str(product.image)
    img = plt.imread(img_path)
    new_image = cv2.Laplacian(img,cv2.CV_64F)
    pat = "media/media/abc.png"
    hist(img,1)
    hist(new_image, 0)
    cv2.imwrite(pat, new_image)
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Laplacian filter"})

def dis(img, u, v):
  i,j = img.shape
  i=i/2
  j=j/2
  return ((u-i)**2+(v-j)**2)*0.5

def ideal_lp_transform(img,radius):
  img = img.astype('float64')
  for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
      if((i+j)%2 != 0):
        img[i][j] = img[i][j] * (-1)

  img_fourier = np.fft.fft2(img)

  mask = np.zeros(img_fourier.shape)

  r =radius
  for i in range(0, img_fourier.shape[0]):
    for j in range(0, img_fourier.shape[1]):
      if(dis(img_fourier,i,j)<r):
        mask[i][j] = 1

  fi = img_fourier*mask
  inverse_ffi = np.fft.ifft2(fi)

  for i in range(0, inverse_ffi.shape[0]):
    for j in range(0, inverse_ffi.shape[1]):
      if((i+j)%2 != 0):
        inverse_ffi[i][j] = inverse_ffi[i][j] * (-1)

  return abs(inverse_ffi)
 
def ideal_lowpass(request):
    if request.method == 'POST':
        gamf = idealForm(request.POST,request.FILES)
        if gamf.is_valid():
            #gamf.save()
            stud = images.objects.last()
            primary = stud.pk
            product = get_object_or_404(images, pk=primary)
            img_path = 'media/'+str(product.image)
            img = plt.imread(img_path)
            rad = request.POST["radius"]
            gam_img = ideal_lp_transform(img,int(rad))
            pat = "media/media/abc.png"
            cv2.imwrite(pat, gam_img)
            hist(img,1)
            hist(gam_img, 0)
            return redirect('/show_ideal_low_pass')
    else:
        gamf = idealForm()
    return render(request,'gamma.html',{'gamf':gamf})

def show_ideal_low_pass(request):
    stud = images.objects.last()
    if request.method == 'POST':
        return redirect('/upload')
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Ideal low pass filter"})

def ideal_highpass(request):
    if request.method == 'POST':
        gamf = idealForm(request.POST,request.FILES)
        if gamf.is_valid():
            #gamf.save()
            stud = images.objects.last()
            primary = stud.pk
            product = get_object_or_404(images, pk=primary)
            img_path = 'media/'+str(product.image)
            img = plt.imread(img_path)
            rad = request.POST["radius"]
            gam_img = ideal_hp_transform(img,int(rad))
            pat = "media/media/abc.png"
            hist(img,1)
            hist(gam_img, 0)
            cv2.imwrite(pat, gam_img)
            return redirect('/show_ideal_high_pass')
    else:
        gamf = idealForm()
    return render(request,'gamma.html',{'gamf':gamf})

def ideal_hp_transform(img,radius):
  img = img.astype('float64')
  for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
      if((i+j)%2 != 0):
        img[i][j] = img[i][j] * (-1)

  img_fourier = np.fft.fft2(img)

  mask = np.zeros(img_fourier.shape)

  r =radius
  for i in range(0, img_fourier.shape[0]):
    for j in range(0, img_fourier.shape[1]):
      if(dis(img_fourier,i,j)>r):
        mask[i][j] = 1

  fi = img_fourier*mask
  inverse_ffi = np.fft.ifft2(fi)

  for i in range(0, inverse_ffi.shape[0]):
    for j in range(0, inverse_ffi.shape[1]):
      if((i+j)%2 != 0):
        inverse_ffi[i][j] = inverse_ffi[i][j] * (-1)

  return abs(inverse_ffi)

def show_ideal_high_pass(request):
    stud = images.objects.last()
    if request.method == 'POST':
        return redirect('/upload')
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Ideal high pass filter"})

def Butterworth(img,u,v, r, n):
  temp = 1/(1+(dis(img, u, v)/r)**(2*n))
  return temp

def butterworth_hp_transform(img, radius, n):
  img = img.astype('float64')
  for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
      if((i+j)%2 != 0):
        img[i][j] = img[i][j] * (-1)

  img_fourier = np.fft.fft2(img)

  mask = np.zeros(img_fourier.shape)

  r = radius

  for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
      mask[i][j] = 1- Butterworth(img,i,j,r, n)

  fi = img_fourier*mask
  inverse_ffi = np.fft.ifft2(fi)

  for i in range(0, inverse_ffi.shape[0]):
    for j in range(0, inverse_ffi.shape[1]):
      if((i+j)%2 != 0):
        inverse_ffi[i][j] = inverse_ffi[i][j] * (-1)

  return abs(inverse_ffi)

def butterworth_lp_transform(img, radius, n):
  img = img.astype('float64')
  for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
      if((i+j)%2 != 0):
        img[i][j] = img[i][j] * (-1)

  img_fourier = np.fft.fft2(img)

  mask = np.zeros(img_fourier.shape)

  r = radius

  for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]):
      mask[i][j] = Butterworth(img,i,j,r, n)

  fi = img_fourier*mask
  inverse_ffi = np.fft.ifft2(fi)

  for i in range(0, inverse_ffi.shape[0]):
    for j in range(0, inverse_ffi.shape[1]):
      if((i+j)%2 != 0):
        inverse_ffi[i][j] = inverse_ffi[i][j] * (-1)

  return abs(inverse_ffi)

def butterworth_lowpass(request):
    if request.method == 'POST':
        gamf = butterworthForm(request.POST,request.FILES)
        if gamf.is_valid():
            stud = images.objects.last()
            primary = stud.pk
            product = get_object_or_404(images, pk=primary)
            img_path = 'media/'+str(product.image)
            img = plt.imread(img_path)
            rad = request.POST["radius"]
            n = request.POST["n"] 
            gam_img = butterworth_lp_transform(img,int(rad),float(n))
            pat = "media/media/abc.png"
            hist(img,1)
            hist(gam_img, 0)
            cv2.imwrite(pat, gam_img)
            return redirect('/show_butterworth_low_pass')
    else:
        gamf = butterworthForm()
    return render(request,'gamma.html',{'gamf':gamf})

def show_butterworth_low_pass(request):
    stud = images.objects.last()
    if request.method == 'POST':
        return redirect('/upload')
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Butterworth lowpass filter"})

def butterworth_highpass(request):
    if request.method == 'POST':
        gamf = butterworthForm(request.POST,request.FILES)
        if gamf.is_valid():
            #gamf.save()
            stud = images.objects.last()
            primary = stud.pk
            product = get_object_or_404(images, pk=primary)
            img_path = 'media/'+str(product.image)
            img = plt.imread(img_path)
            rad = request.POST["radius"]
            n = request.POST["n"]
            gam_img = butterworth_hp_transform(img,int(rad),float(n))
            pat = "media/media/abc.png"
            hist(img,1)
            hist(gam_img, 0)
            cv2.imwrite(pat, gam_img)
            return redirect('/show_butterworth_high_pass')
    else:
        gamf = butterworthForm()
    return render(request,'gamma.html',{'gamf':gamf})

def show_butterworth_high_pass(request):
    stud = images.objects.last()
    if request.method == 'POST':
        return redirect('/upload')
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Butterworth highpass filter"})

def max(request):
    if request.method == 'POST':
        return redirect('/upload')
    stud = images.objects.last()
    primary = stud.pk
    product = get_object_or_404(images, pk=primary)
    img_path = 'media/'+str(product.image)
    img = plt.imread(img_path)
    log_img = max_filter(img)
    pat = "media/media/abc.png"
    hist(img,1)
    hist(log_img, 0)
    cv2.imwrite(pat, log_img)
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Max filter"})

def max_filter(im1):
    im1 =   Image.fromarray(im1)
    im2 = im1.filter(ImageFilter.MaxFilter(size = 3)) 
    im2 = np.array(im2)
    return im2

def min(request):
    if request.method == 'POST':
        return redirect('/upload')
    stud = images.objects.last()
    primary = stud.pk
    product = get_object_or_404(images, pk=primary)
    img_path = 'media/'+str(product.image)
    img = plt.imread(img_path)
    log_img = max_filter(img)
    pat = "media/media/abc.png"
    hist(img,1)
    hist(log_img, 0)
    cv2.imwrite(pat, log_img)
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Min filter"})


def erode_dilate(request):
    if request.method == 'POST':
        return redirect('/upload')
    stud = images.objects.last()
    primary = stud.pk
    product = get_object_or_404(images, pk=primary)
    img_path = 'media/'+str(product.image)
    img = plt.imread(img_path)
    log_img = erode_filter(img)
    log_img = dilate_filter(log_img)
    pat = "media/media/abc.png"
    hist(img,1)
    hist(log_img, 0)
    cv2.imwrite(pat, log_img)
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Min filter"})

def min_filter(im1):
    im1 =   Image.fromarray(im1)
    im2 = im1.filter(ImageFilter.MinFilter(size = 3)) 
    im2 = np.array(im2)
    return im2

def erode(request):
    if request.method == 'POST':
        return redirect('/upload')
    stud = images.objects.last()
    primary = stud.pk
    product = get_object_or_404(images, pk=primary)
    img_path = 'media/'+str(product.image)
    img = plt.imread(img_path)
    log_img = erode_filter(img)
    pat = "media/media/abc.png"
    hist(img,1)
    hist(log_img, 0)
    cv2.imwrite(pat, log_img)
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Erode filter"})

def erode_filter(im1):
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.erode(im1, kernel)
    return image
    
def dilate(request):
    if request.method == 'POST':
        return redirect('/upload')
    stud = images.objects.last()
    primary = stud.pk
    product = get_object_or_404(images, pk=primary)
    img_path = 'media/'+str(product.image)
    img = plt.imread(img_path)
    log_img = dilate_filter(img)
    hist(img,1)
    hist(log_img,0)
    pat = "media/media/abc.png"
    cv2.imwrite(pat, log_img)
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Dilate filter"})

def dilate_filter(im1):
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(im1, kernel, iterations=1)
    return img_dilation

def hist(image, original):
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 255))
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    if original:
        pat = "media/media/hist.png"
    else:
        pat = "media/media/hist1.png"
    h = np.histogram(image, 255)
    plt.plot(h[0])
    plt.savefig(pat)
    return histogram

def mean(request):
    if request.method == 'POST':
        return redirect('/upload')
    stud = images.objects.last()
    print("------------------------",stud.pk)
    primary = stud.pk
    product = get_object_or_404(images, pk=primary)
    img_path = 'media/'+str(product.image)
    # img = plt.imread(img_path)
    img = cv2.imread(img_path)
    figure_size = 5 # the dimension of the x and y axis of the kernal.
    new_image = cv2.blur(img,(figure_size, figure_size))
    pat = "media/media/abc.png"
    hist(img,1)
    hist(new_image, 0)
    cv2.imwrite(pat, new_image)
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Mean filter"})

def img2sketch(photo, k_size = 7):
    img=cv2.imread(photo)
    grey_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #invert_img=cv2.bitwise_not(grey_img)
    blur_img=cv2.GaussianBlur(grey_img, (k_size,k_size),0)
    #invblur_img=cv2.bitwise_not(blur_img)
    sketch_img=cv2.divide(grey_img,blur_img, scale=256.0)
    return sketch_img

def sketch(request):
    if request.method == 'POST':
        gamf = kernelForm(request.POST,request.FILES)
        if gamf.is_valid():
            #gamf.save()
            stud = images.objects.last()
            primary = stud.pk
            product = get_object_or_404(images, pk=primary)
            img_path = 'media/'+str(product.image)
            img = plt.imread(img_path)
            temp = request.POST["kernel"]
            gam_img = img2sketch(img_path,int(temp))
            pat = "media/media/abc.png"
            cv2.imwrite(pat, gam_img)
            hist(img,1)
            hist(gam_img, 0)
            return redirect('/show_sketch')
    else:
        gamf = kernelForm()
    return render(request,'gamma.html',{'gamf':gamf})

def show_sketch(request):
    stud = images.objects.last()
    if request.method == 'POST':
        return redirect('/upload')
    return render(request,'show_posts.html',{'img':stud, 'media_url':settings.MEDIA_URL, 'function': "Gamma transformation"})