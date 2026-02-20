import os
from celery import Celery

redis_url = os.getenv("REDIS_URL")

celery = Celery(
    "tune_trainer",
    broker=redis_url,
    backend=redis_url
)

@celery.task
def ping():
    return "still alive"
