"""innovative URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from editor.views import *
from django.conf.urls.static import static


urlpatterns = [
    path("admin/", admin.site.urls),
    path('account/', include('accounts.urls')),
    path("upload",upload_image),
    path("gamma",gamma),
    path("show_gamma",show_gamma),
    path("show_log",show_log),
    path("median",median),
    path("max",max),
    path("erode_dilate",erode_dilate),
    path("min",min),
    path("mean",mean),
    path("sketch",sketch),
    path("show_sketch",show_sketch),
    path("erode",erode),
    path("dilate",dilate),
    path("mode",mode),
    path("contrast",contrast_stretching),
    path("show_contrast",show_contrast),
    path("show_Laplacian",show_Laplacian),
    path("show_histogram_equalization",show_histogram_equalization),   

    path("ideal_lowpass",ideal_lowpass),
    path("show_ideal_low_pass",show_ideal_low_pass),
    path("ideal_highpass",ideal_highpass),
    path("show_ideal_high_pass",show_ideal_high_pass),
    path("butterworth_lowpass",butterworth_lowpass),
    path("show_butterworth_low_pass",show_butterworth_low_pass),
    path("butterworth_highpass",butterworth_highpass),
    path("show_butterworth_high_pass",show_butterworth_high_pass),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)