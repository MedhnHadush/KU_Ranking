from django.urls import path
from . import views

app_name = 'ranking_analyzer'

urlpatterns = [
    path('', views.home, name='home'),
    path('calculate/', views.calculate_ranking, name='calculate_ranking'),
] 