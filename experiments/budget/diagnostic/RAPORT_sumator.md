# Rozbieżność modelu dla sumatora kontrolowanego (test lustrzany, ibm_marrakesh, 01.10.2026)

Dane: zadanie davcn5o4oijs73e7b5m0 (opcje domyślne SamplerV2: bez DD i twirlingu; 15 obwodów w 19 s). Obwody ISA z job.inputs, kalibracja z chwili zadania. Skrypty: `symulacja.py` (dokładna macierz gęstości, ALAP, relaksacja, błędy bramek, koherentny ZZ), `hipotezy.py`.

## 1. Gdzie jest rozbieżność
| wariant | zmierzone P0 | model analityczny (bezczynność+ZZ+λ, LOO) | symulacja bazowa |
|---|---|---|---|
| obwód dokładny (ref) | 0,149 | 0,242 | 0,339 |
| pass | 0,308 | 0,321 | 0,413 |
| count-greedy | 0,134 | 0,116 | 0,147 |
Rozbieżność dotyczy głównie wariantu ref; pass i greedy są dobrze przewidziane.

## 2. Sygnatura w danych
- Bit 1 (logiczny a[0], mierzony na q2) jest w ref prawie losowy: P(1)=0,47 (symulacja 0,22). Najczęstszy wynik ref to `0000010` (15,5%), nie `0000000` (14,9%).
- Nadmiar na bicie 1 występuje we wszystkich trzech wariantach i rośnie z długością obwodu: greedy +0,11 (10,9 µs), pass +0,14 (12,1 µs), ref +0,25 (15,8 µs).
- W greedy wszystkie pozostałe bity zgadzają się z symulacją bazową co do 0,01–0,03 — błąd jest zlokalizowany.
- Ref i greedy kończą na podobnym „dnie” (0,149 i 0,134), choć ref jest o ~5 µs dłuższy: zachowanie nasycające, nie addytywne.

## 3. Hipotezy (symulacja wszystkich 15 obwodów; miara SS = Σ ln²(hw/sim))
| hipoteza | SS | sumator ref/pass/greedy |
|---|---|---|
| bazowa | 1,01 | 0,339 / 0,413 / 0,147 |
| q16 T2=10 µs | 0,48 | 0,237 / 0,311 / 0,114 |
| koherentna faza CZ 0,05 rad | 0,40 | 0,178 / 0,287 / 0,091 |
| nadmiarowa depolaryzacja CZ 4e-3 | 0,40 | 0,203 / 0,259 / 0,100 |
| q2 T2=20 µs | 0,59 | bit 1 pass/greedy odtworzony (0,33/0,33), ref 0,37 vs 0,47 |
| λ 2e-3 + faza 0,03 | 0,37 | 0,193 / 0,278 / 0,096 |
Żadna kombinacja błędów markowowskich zgodnych z kalibracją nie godzi jednocześnie ref (0,149) i greedy (0,134): to, co ściąga ref, ściąga greedy do 0,08–0,10 i psuje half_uncomputed (te same kubity 2–6, krótszy).

## 4. Kalibracja jest nieaktualna dla T1/T2
T1 i T2 wszystkich użytych kubitów są identyczne w kalibracji z 01.10 22:53 i 02.10 12:43 (np. q2 347/399 µs, q16 98/37 µs), podczas gdy błędy odczytu i CZ się zmieniły (krawędź (3,16): 1,7e-3 → 3,4e-3). Czasy koherencji nie są mierzone codziennie, a fluktuują w skali godzin (TLS). Model dostaje więc T1/T2 sprzed nieznanego czasu.

## 5. Wyjaśnienie (najbardziej prawdopodobne)
Rozbieżność składa się z (i) ogólnego nadmiarowego błędu na CZ, widocznego też w obwodach nested (λ≈2e-3, zgodnie z dopasowaniem LOO), i (ii) **zlokalizowanego, silnego defektu koherencji na drodze logicznego kubitu a[0], kończącej się na q2**, nieobecnego w (nieaktualnej) kalibracji T1/T2. Defekt rośnie z czasem trwania i nasyca bit 1 w najdłuższym wariancie (ref), stąd największy błąd modelu właśnie tam. To jest własność urządzenia w chwili zadania, nie błąd modelu budżetu ani passu; kierunek różnicy pass vs ref jest przewidziany poprawnie.

## 6. Test rozstrzygający (wymaga QPU, nie wykonany)
- powtórzenie 3 wariantów sumatora na tym samym układzie (zmienność w czasie),
- te same 3 warianty na układzie omijającym q2 i q16,
- pomiar T1 i echa Hahna (T2) na q2, q16 i sąsiadach.
Szacunek: ~13 s QPU (zostało 125 s na us-east-open).

## 7. Test na IBM (02.10.2026, zadanie davpkglj371s73dnb7hg, 11 s QPU; zostało 15 s)
**T1 i echo Hahna zmierzone vs kalibracja (µs):**
| kubit | T1 zmierz. / kal. | T2 echo zmierz. / kal. |
|---|---|---|
| q2 (kończy bit 1) | **40 / 347** | **27 / 399** |
| q6 | **80 / 321** | **50 / 181** |
| q3 | 225 / 188 | 77 / 194 |
| q16 | 144 / 98 | 32 / 37 |
**A. Ten sam układ (obwody ISA z 01.10):** ref 0,136, pass 0,267, greedy 0,101 (01.10: 0,149 / 0,308 / 0,134); bit 1 nadal 0,38–0,46 — efekt powtarzalny.
**B. Inny obszar (kubity 32–54):** ref 0,175, pass 0,190, greedy 0,095; bit 1 też ~0,41–0,46 — a[0] jest najbardziej narażonym kubitem logicznym obwodu (najwięcej bramek i oczekiwania), nie tylko ofiarą q2.
**Symulacja ze zmierzonymi T1/T2 (q2,q16,q3,q6):** pass 0,289 (hw 0,267), greedy 0,107 (hw 0,101), ref 0,219 (hw 0,136); bit 1 w symulacji 0,31–0,35. Z dodatkową koherentną fazą CZ 0,02 rad: pass 0,267 (=hw), greedy 0,094, ref 0,190.

## 8. Wniosek
Główna przyczyna: **nieaktualna kalibracja koherencji**. Kubit q2 ma w rzeczywistości ~9× krótsze T1 i ~15× krótsze T2 niż podaje kalibracja, q6 ~4× krótsze T1. Po podstawieniu zmierzonych wartości model odtwarza pass i count-greedy. Reszta dla obwodu dokładnego (0,19 vs 0,14) jest zgodna z koherentnym błędem fazy CZ, który w E†E dodaje się liniowo, i z niezmierzoną koherencją q4, q5, q7, które ten wariant obciąża najdłużej. Pass i model budżetu nie są przyczyną; przewaga passu nad obwodem dokładnym utrzymuje się w obu obszarach (A: 0,267 vs 0,136; B: 0,190 vs 0,175).
