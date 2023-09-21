from django import forms
from .models import *
class imageForm(forms.ModelForm):
    class Meta:
        model = images
        fields = ['image','filters']
        #widgets = {'category':forms.Select(attrs={'onchange':'gamma()'})}

class gammaForm(forms.ModelForm):
    class Meta:
        model = images
        fields = ['gamma']

class kernelForm(forms.ModelForm):
    class Meta:
        model = images
        fields = ['kernel']

class contrastForm(forms.ModelForm):
    class Meta:
        model = images
        fields = ['r1','s1','r2','s2']

class idealForm(forms.ModelForm):
    class Meta:
        model = images
        fields = ['radius']

class butterworthForm(forms.ModelForm):
    class Meta:
        model = images
        fields = ['radius','n']