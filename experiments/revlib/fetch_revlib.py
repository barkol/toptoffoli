"""Pobiera pliki .real wymienione w wybor.json (i wszystkie z listy realizacji MCT) z revlib.org do real/."""
import json, os, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(HERE, "real"), exist_ok=True)
for o in json.load(open(os.path.join(HERE, "wybor.json"))) + json.load(open(os.path.join(HERE, "nieuruchomione.json"))):
    p = os.path.join(HERE, "real", o["file"])
    if not os.path.exists(p):
        subprocess.run(["curl", "-s", "-L", "-o", p, "https://www.revlib.org/doc/real/" + o["file"]], check=True); time.sleep(0.3)
