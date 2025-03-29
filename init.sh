#!/bin/bash

pip install langchain_core langchain_ollama langchain_openai psutil

nohup ollama serve > /dev/null 2>&1 &
