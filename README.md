# Audiopet

Web app for an Audiopet companion, which learns to sing melodies based on user input.

Uses Aubio to detect user input (digital signal processing), no network connection needed. 

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Web app usage

```bash
python3 app.py
```

Then open http://127.0.0.1:5000 in your browser. Select a character, click Record Input to record user input (humming, singing etc.) and Stop Recording to submit the input. You can listen to the latest Audiopet response using Play Last Reply.
