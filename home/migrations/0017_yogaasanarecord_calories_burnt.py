# Generated by Django 4.2.3 on 2023-08-09 14:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("home", "0016_alter_yogaasanarecord_aasna"),
    ]

    operations = [
        migrations.AddField(
            model_name="yogaasanarecord",
            name="calories_burnt",
            field=models.DecimalField(decimal_places=2, default=0, max_digits=6),
        ),
    ]
