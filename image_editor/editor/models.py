from django.db import models

# Create your models here.
class users(models.Model):
    name = models

class images(models.Model):
    #user = models.ForeignKey(users.name,on_delete=models.CASCADE)
    image = models.ImageField(upload_to='media')
    COLOR_CHOICES = (
    ('gamma','gamma'),
    ('log', 'log'),
    ('histogram','histogram'),
    ('contrast','contrast'),
    ('median','median'),
    ('max','max'),
    ('min','min'),
    ('erode','erode'),
    ('dilate','dilate'),
    ('mode','mode'),
    ('mean','mean'),
    ('Laplacian','Laplacian'),
    ('Ideal_low_Pass','Ideal Low Pass'),
    ('Ideal_high_Pass','Ideal high Pass'),
    ('Butterworth_low_Pass','Butterworth low Pass'),
    ('Butterworth_high_Pass','Butterworth high Pass'),
    ('sketch','Sketch'),
    ('erode_dilate','erode_dilate'),
    )
    filters = models.CharField(choices=COLOR_CHOICES, default='gamma',max_length=23)  
    gamma = models.FloatField(default=3)
    kernel = models.FloatField(default=3)
    r1 = models.IntegerField(default=0)
    s1 = models.IntegerField(default=0)
    r2 = models.IntegerField(default=0)
    s2 = models.IntegerField(default=0)
    radius = models.IntegerField(default=40)
    n = models.FloatField(default=0.1)



