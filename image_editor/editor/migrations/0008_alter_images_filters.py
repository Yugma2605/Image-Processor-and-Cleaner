# Generated by Django 4.1 on 2022-11-25 04:55

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("editor", "0007_images_kernel"),
    ]

    operations = [
        migrations.AlterField(
            model_name="images",
            name="filters",
            field=models.CharField(
                choices=[
                    ("gamma", "gamma"),
                    ("log", "log"),
                    ("histogram", "histogram"),
                    ("contrast", "contrast"),
                    ("median", "median"),
                    ("max", "max"),
                    ("min", "min"),
                    ("erode", "erode"),
                    ("dilate", "dilate"),
                    ("mode", "mode"),
                    ("mean", "mean"),
                    ("Laplacian", "Laplacian"),
                    ("Ideal_low_Pass", "Ideal Low Pass"),
                    ("Ideal_high_Pass", "Ideal high Pass"),
                    ("Butterworth_low_Pass", "Butterworth low Pass"),
                    ("Butterworth_high_Pass", "Butterworth high Pass"),
                    ("sketch", "Sketch"),
                    ("erode_dilate", "erode_dilate"),
                ],
                default="gamma",
                max_length=23,
            ),
        ),
    ]