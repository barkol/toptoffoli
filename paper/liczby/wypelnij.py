"""Wypelnia znaczniki @@KLUCZ@@ z liczby/liczby.json. szablon -> wynik. Przerywa na brakujacym kluczu."""
import json, re, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
K = json.load(open(os.path.join(HERE, "liczby.json")))
for src, dst in zip(sys.argv[1::2], sys.argv[2::2]):
    s = open(src).read(); miss = sorted({k for k in re.findall(r"@@(\w+)@@", s) if k not in K})
    if miss: print("BRAK kluczy w", src, miss); sys.exit(1)
    open(dst, "w").write(re.sub(r"@@(\w+)@@", lambda m: str(K[m.group(1)]), s)); print("ok", dst)
