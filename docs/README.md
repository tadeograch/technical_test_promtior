# Promtior chatbot

A chatbot assistant that uses the RAG architecture to answer questions about the
content of the Promtior website, based on the LangChain library.

## Technologies
This chatbot was built in python, using LangChain framework and OpenAI model. Based on [LangChain documentation](https://python.langchain.com/v0.1/docs/get_started/quickstart/).
## Architecture
<img src="https://github.com/tadeograch/technical_test_promtior/blob/test/docs/promtior_chatbot_diagram.png/">
## Installation

Clone the repository of the Promtior chatbot project
```bash
git clone https://github.com/tadeograch/technical_test_promtior.git
```
Cd into promptior chatbot directory
```bash
cd promptior_chatbot
```
Install requirements
```bash
pip install -r requirements.txt
```
## Basic usage

If you want to run the chatbot locally:
```bash
python serve.py
```
And to ask something you can run the client.py with your custom question, for example:
```bash
python client.py "What services does Promtior offer?"
```
Or you can interact with LangServe Playground:
```bash
http://localhost:8000/promtior_chatbot/playground
```
## Overview

This project was created as a technical test, presenting a new challenge for someone unfamiliar with Retrieval Augmented Generation. Learning something new is never easy, but I know this is only the beginning of the journey.

The exciting part of this project was starting to understand the power of AI and all that it can do. I hope this experience not only helps me secure a job but also inspires my future in the software engineering industry. 
## Authors

Tadeo Grach