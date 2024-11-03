# chatbot-gen-AI1

Loaders
1. Text Loader

## Create a virtual environment :

(Windows)
```
python -m venv env
```

(MacOS)
```
python3.11 -m venv env
```

## Activate the virtual environment :

```
source env/bin/activate
```

## Installation:
(Windows)
```
pip install -r requirements.txt
pip install langchainhub
```

(MacOS)

```
pip3 install -r requirements.txt
```

Map Reduce

```
sending each relative chunk to LLM rather sending whole relative chunk to LLM which may well exceed the 
token limitation.
from the retrived answer , once again LLM called to get best answer or combined answer.
```