"""
celery.py
=========
Celery application configuration.
This file is imported by Django workers and the Celery worker process.
"""
 
import os
from celery import Celery
 
# Tell Celery which Django settings to use
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
 
app = Celery("rag_web")
 
# Load Celery config from Django settings (CELERY_ prefix)
app.config_from_object("django.conf:settings", namespace="CELERY")
 
# Auto-discover tasks in all installed apps
# Looks for rag_web/tasks.py automatically
app.autodiscover_tasks()