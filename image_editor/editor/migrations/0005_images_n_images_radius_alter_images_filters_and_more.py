# Generated by Django 4.1 on 2022-11-24 06:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("editor", "0004_alter_images_filters"),
    ]

    operations = [
        migrations.AddField(
            model_name="images", name="n", field=models.FloatField(default=0.1),
        ),
        migrations.AddField(
            model_name="images", name="radius", field=models.IntegerField(default=40),
        ),
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
                    ("Laplacian", "Laplacian"),
                    ("Ideal_low_Pass", "Ideal Low Pass"),
                    ("Ideal_high_Pass", "Ideal high Pass"),
                    ("Butterworth_low_Pass", "Butterworth low Pass"),
                    ("Butterworth_high_Pass", "Butterworth high Pass"),
                ],
                default="gamma",
                max_length=23,
            ),
        ),
        migrations.AlterField(
            model_name="images", name="gamma", field=models.FloatField(default=3),
        ),
    ]
