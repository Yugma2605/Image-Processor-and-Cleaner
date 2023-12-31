# Generated by Django 4.1 on 2022-11-23 19:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("editor", "0002_images_r1_images_r2_images_s1_images_s2_and_more"),
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
                    ("mode", "mode"),
                    ("mean", "mean"),
                ],
                default="gamma",
                max_length=23,
            ),
        ),
        migrations.AlterField(
            model_name="images", name="gamma", field=models.IntegerField(default=3),
        ),
    ]
