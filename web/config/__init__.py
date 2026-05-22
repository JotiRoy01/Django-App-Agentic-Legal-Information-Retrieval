# This makes Celery load when Django starts
# Required for @shared_task decorator to work in tasks.py
from .celery import app as celery_app
 
__all__ = ("celery_app",)