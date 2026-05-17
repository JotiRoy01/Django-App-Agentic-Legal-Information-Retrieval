from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from agentic.pipeline.rag_pipeline import RAGPipeline


def index(request):
    return HttpResponse("Hello, world. You're at the rag_web index.")