from django.contrib import admin

from .models import RegisteredUser, CurlRecord, SquatRecord, AasnaVideo, PushupRecord, YogaAsanaRecord

admin.site.register(RegisteredUser)
admin.site.register(CurlRecord)
admin.site.register(SquatRecord)
admin.site.register(AasnaVideo)
admin.site.register(YogaAsanaRecord)
admin.site.register(PushupRecord)
