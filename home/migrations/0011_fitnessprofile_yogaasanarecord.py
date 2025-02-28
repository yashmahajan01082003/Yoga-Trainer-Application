# Generated by Django 4.2.3 on 2023-08-09 08:18

from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ("home", "0010_alter_aasnavideo_video_file"),
    ]

    operations = [
        migrations.CreateModel(
            name="FitnessProfile",
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
                ("date", models.DateField(default=django.utils.timezone.now)),
                (
                    "calories_burnt",
                    models.DecimalField(decimal_places=2, default=0, max_digits=6),
                ),
                (
                    "target_calories",
                    models.DecimalField(decimal_places=2, default=0, max_digits=6),
                ),
                ("squats_done", models.PositiveIntegerField(default=0)),
                ("pushups_done", models.PositiveIntegerField(default=0)),
                ("curls_done", models.PositiveIntegerField(default=0)),
                (
                    "user",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="home.registereduser",
                    ),
                ),
            ],
        ),
        migrations.CreateModel(
            name="YogaAsanaRecord",
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
                ("is_done", models.BooleanField(default=False)),
                ("date", models.DateField(default=django.utils.timezone.now)),
                (
                    "aasna",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="home.aasnavideo",
                    ),
                ),
                (
                    "fitness_profile",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        to="home.fitnessprofile",
                    ),
                ),
            ],
        ),
    ]
