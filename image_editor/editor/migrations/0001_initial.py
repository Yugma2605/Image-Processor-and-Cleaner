# Generated by Django 4.1.3 on 2022-11-22 10:13

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="images",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("image", models.ImageField(upload_to="media")),
                (
                    "filters",
                    models.CharField(
                        choices=[
                            ("gamma", "gamma"),
                            ("log", "log"),
                            ("histogram", "histogram"),
                            ("contrast", "contrast"),
                            ("median", "median"),
                        ],
                        default="gamma",
                        max_length=23,
                    ),
                ),
                ("gamma", models.IntegerField(max_length=20)),
            ],
        ),
        migrations.CreateModel(
            name="users",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
            ],
        ),
    ]
