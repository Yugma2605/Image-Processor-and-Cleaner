# Generated by Django 4.1 on 2022-11-25 04:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("editor", "0006_alter_images_filters"),
    ]

    operations = [
        migrations.AddField(
            model_name="images", name="kernel", field=models.FloatField(default=3),
        ),
    ]
