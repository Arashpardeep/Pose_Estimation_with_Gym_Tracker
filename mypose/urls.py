from django.contrib import admin
from django.urls import path
from mypose import views

urlpatterns = [
    path("", views.signin, name = 'signin'),
    path("Signup", views.Signup, name = 'signup'),
    path("index", views.index, name = 'index'),
    path("blog", views.blog, name = 'blog'),
    path("landmarks", views.landmarks, name = 'landmarks'),
    path("signout", views.signout, name = 'signout'),
    path("profile", views.profile, name = 'profile'),
    path("webcam", views.webcam, name = 'webcam'),
    path("bicep", views.bicep, name = 'bicep'),
    path("squat", views.squat, name = 'squat'),
    path("pushup", views.pushup, name = 'pushup'),
    path("lat", views.lat, name = 'lat'),
    path("shoulder", views.shoulder, name = 'shoulder'),
]