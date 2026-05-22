


"""
urls.py — dashboard URL patterns
"""
from django.urls import path
from . import views
 
urlpatterns = [
    path("",                      views.index,      name="index"),
    path("run-query/",            views.run_query,  name="run_query"),
    path("status/<str:task_id>/", views.status,     name="status"),
    path("results/<str:task_id>/",views.results,    name="results"),
    path("evaluation/",           views.evaluation, name="evaluation"),
    path("history/",              views.history,    name="history"),
]
 