from datetime import date
from decimal import Decimal

from django.db import models
from django.utils import timezone

class RegisteredUser(models.Model):
    full_name = models.CharField(max_length=100)
    age = models.PositiveIntegerField(default=0)
    gender = models.CharField(max_length=10)
    username = models.CharField(max_length=50, unique=True)
    password = models.CharField(max_length=100)
    weight = models.DecimalField(max_digits=5, decimal_places=2, default=0)  # Weight in kilograms

    def __str__(self):
        return self.username

class CurlRecord(models.Model):
    user = models.ForeignKey(RegisteredUser, on_delete=models.CASCADE)
    curls_done = models.PositiveIntegerField(default=0)
    date = models.DateField(default=timezone.now)  # Use timezone.now as default value

    def calculate_calories_burned(self):
        # Example formula for calorie calculation:
        # calories_burned = (curls_done * 0.5) * (weight_in_kg) * (0.02) + (age_in_years * 0.1)
        # You should replace this with a scientifically validated formula
        weight_in_kg = self.user.weight
        age_in_years = self.user.age
        calories_burned = (self.curls_done * 0.5) * weight_in_kg * 0.02 + age_in_years * 0.1
        return calories_burned

    def __str__(self):
        return f"User: {self.user.username}, Curls: {self.curls_done}, Date: {self.date}"

class SquatRecord(models.Model):
    user = models.ForeignKey(RegisteredUser, on_delete=models.CASCADE)
    squats_done = models.PositiveIntegerField(default=0)
    date = models.DateField(default=timezone.now)  # Use timezone.now as default value

    def calculate_calories_burned(self):
        # Example formula for calorie calculation for squats:
        # calories_burned = (squats_done * 0.8) * (weight_in_kg) * (0.02) + (age_in_years * 0.1)
        # You should replace this with a scientifically validated formula
        weight_in_kg = self.user.weight
        age_in_years = self.user.age
        calories_burned = (self.squats_done * 0.8) * weight_in_kg * 0.02 + age_in_years * 0.1
        return calories_burned

    def __str__(self):
        return f"User: {self.user.username}, Squats: {self.squats_done}, Date: {self.date}"

class PushupRecord(models.Model):
    user = models.ForeignKey(RegisteredUser, on_delete=models.CASCADE)
    pushups_done = models.PositiveIntegerField(default=0)
    date = models.DateField(default=timezone.now)

    def __str__(self):
        return f"User: {self.user.username}, Push-ups: {self.pushups_done}, Date: {self.date}"


class AasnaVideo(models.Model):
    name = models.CharField(max_length=100)
    video_file = models.ImageField(upload_to='imagesYoga/')
    is_aasana_done = models.BooleanField(default=False)
    caloriesburn = models.PositiveIntegerField(default=0)

    def __str__(self):
        return self.name


class FitnessProfile(models.Model):
    user = models.ForeignKey(RegisteredUser, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.now)  # Date of the fitness record
    calories_burnt = models.DecimalField(max_digits=6, decimal_places=2, default=0)  # Calories burnt
    target_calories = models.DecimalField(max_digits=6, decimal_places=2, default=0)  # Target calories for the day
    squats_done = models.PositiveIntegerField(default=0)  # Number of squats done
    pushups_done = models.PositiveIntegerField(default=0)  # Number of pushups done
    curls_done = models.PositiveIntegerField(default=0)  # Number of curls done

    def calculate_calories_burned(self):
        # Calculate calories burnt based on the exercises done (similar to previous models)
        CurlRecord1 = CurlRecord.objects.filter(user=self.user,date=self.date).first()
        self.curls_done = CurlRecord1.curls_done

        # Calculate calories burnt based on the exercises done (similar to previous models)
        SquatRecord1 = SquatRecord.objects.filter(user=self.user, date=self.date).first()
        self.squats_done = SquatRecord1.squats_done

        # Calculate calories burnt based on the exercises done (similar to previous models)
        PushupRecord1 = PushupRecord.objects.filter(user=self.user, date=self.date).first()
        self.pushups_done = PushupRecord1.pushups_done

        today = date.today()
        FitnessProfile1 = FitnessProfile.objects.filter(user=self.user,date=today).first()
        YogaAsanaRecord1 = YogaAsanaRecord.objects.filter(fitness_profile=FitnessProfile1,date=today)
        lenY=0
        for yoga in YogaAsanaRecord1:
            lenY += yoga.times
        lenY *= Decimal(2.5)
        weight_in_kg = self.user.weight
        age_in_years = self.user.age
        # Convert constants to Decimal
        squat_constant = Decimal('0.7')
        pushup_constant = Decimal('0.6')
        curl_constant = Decimal('0.5')
        weight_constant = Decimal('0.02')
        age_constant = Decimal('0.1')

        # Calculate actual calories burned based on exercises done
        actual_calories_burned = (
                (
                            self.squats_done * squat_constant + self.pushups_done * pushup_constant + self.curls_done * curl_constant)
                * weight_in_kg * weight_constant
                + age_in_years * age_constant + lenY
        )

        # Calculate target calories based on weight and age
        target_calories = (weight_in_kg * Decimal('25')) + (age_in_years * Decimal('5'))

        self.target_calories=target_calories
        self.calories_burnt = actual_calories_burned
        caloriesburn = [self.curls_done*curl_constant * weight_in_kg*weight_constant ,self.squats_done*squat_constant * weight_in_kg*weight_constant,self.pushups_done*pushup_constant* weight_in_kg*weight_constant]
        return caloriesburn

    def __str__(self):
        return f"User: {self.user.username}, Date: {self.date}"


class YogaAsanaRecord(models.Model):
    fitness_profile = models.ForeignKey(FitnessProfile, on_delete=models.CASCADE)
    aasna = models.CharField(default='',max_length=400)
    calories_burnt = models.DecimalField(max_digits=6, decimal_places=2, default=0)  # Calories burnt
    is_done = models.BooleanField(default=False)
    times = models.PositiveIntegerField(default=1)
    date = models.DateField(default=timezone.now)  # Date when the yoga asana was done

    def comp(self):
        AasnaVideo1 = AasnaVideo.objects.all()
        for av in AasnaVideo1:
            if self.aasna.lower() == av.name:
                self.calories_burnt = av.caloriesburn
                av.is_done=True
                av.save()
        return "hello"


    def __str__(self):
        return f"User: {self.fitness_profile.user.username}, Aasana: {self.aasna}, Date: {self.date}"

