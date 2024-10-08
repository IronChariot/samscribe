Small script which lets you hold down a keyboard key, talk, release the key, and then after the whisper model has transcribed the speech to text, it'll be put into your clipboard ready for pasting, and a small 'beep' will sound to let you know this has finished.

Before this can run, you'll need:
```
pip install --upgrade keyboard sounddevice scipy torch transformers accelerate pyperclip pygame
```

(Probably more, will add to this later)
