# Django-App-Agentic-Legal-Information-Retrieval
This is the kaggle competition problem I build this top of RAG architecture. Now I am build a web app that my RAG application interact with web application

# create environment
```bash
source activate base  
conda create --prefix ./django_rag python=3.10 -y  
conda activate ./django_rag
```
# install pyproject
```bash
pip install -e .
```
# start with Django
create config folder inside the web
```bash
django-admin startproject config web
```
# Install the rag_web application
>> Django automaticaly create the rag_web app folder. you don't initiate first
```bash
python manage.py startapp rag_web
```
# then instll app add the setting.py files
```bash
'rag_web'
```
# After initialize the models for DB
```bash
python manage.py makemigrations rag_web
python manage.py migrate
```
# Run the tailwind
```bash
python manage.py tailwind start
```
# Run the celery
```bash
cd web
celery -A config worker --loglevel=info --pool=solo
```
# Run the redis
```bash
# If installed as Windows service, start it:
net start Redis
```
# Run the redis on the powersheel
```bash
docker start my-redis
```
# Or remove the redis from the docker then
```bash
docker stop my-redis
docker rm my-redis
docker run -d -p 6379:6379 --name my-redis redis
```
# Activate conda environment in antigravity
```bash
source /c/anaconda3/etc/profile.d/conda.sh
conda activate django_rag
```