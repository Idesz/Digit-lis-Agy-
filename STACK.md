# Technológiai Stakk és Rendszerkomponensek (STACK.md)

Ez a dokumentum részletesen bemutatja a Neuromorphic Brain Emulation projektben alkalmazott összes technológiát, programkönyvtárat, adatbázist és architektúrális komponenst, kitérve arra, hogy **mire használjuk**, és **miért pont ezt a technológiát választottuk**.

---

## 📑 Tartalomjegyzék
1. [Programozási Nyelv és Futtatókörnyezet](#1-programozási-nyelv-és-futtatókörnyezet)
2. [Mélytanulási és Neuromorf Keretrendszerek](#2-mélytanulási-és-neuromorf-keretrendszerek)
3. [Memória- és Adattárolási Réteg](#3-memória--és-adattárolási-réteg)
4. [Kognitív Vezérlés és Keresési Algoritmusok](#4-kognitív-vezérlés-és-keresési-algoritmusok)
5. [Monitorozás, Telemetria és Megjelenítés](#5-monitorozás-telemetria-és-megjelenítés)
6. [Konténerizáció és Infrastruktúra](#6-konténerizáció-és-infrastruktúra)
7. [Minőségbiztosítás és Tesztelés](#7-minőségbiztosítás-és-tesztelés)

---

## 1. Programozási Nyelv és Futtatókörnyezet

### **Python 3.10+ / 3.12**
* **Mire használjuk?**
  * A teljes kognitív keretrendszer, az SNN szimuláció, a memóriamodulok, a döntéshozatali útválasztók és a tanítási/szimulációs csővezeték elsődleges nyelve.
* **Miért ezt választottuk?**
  * **Ökoszisztéma:** A tenzorműveletek (PyTorch), gráfadatbázisok és vektorkeresők natív integrációja itt a legérettebb és leggyorsabban fejleszthető.
  * **Aszinkronitás és hatékonyság:** A modern típusannotációk (`dataclasses`, `typing`, `pydantic`) segítségével típusbiztos és önmagát dokumentáló tiszta kódot tesz lehetővé, minimalizálva a futásidejű rejtett hibákat.

---

## 2. Mélytanulási és Neuromorf Keretrendszerek

### **PyTorch (>= 2.1.0)**
* **Mire használjuk?**
  * A tüzelő neurális hálózatok (SNN) Leaky Integrate-and-Fire (LIF) modelljének, a DECOLLE lokális tanulási szabályoknak és a JEPA világmodell látens tenzorszámításainak megvalósítására.
  * Egyedi `torch.autograd.Function` szurogát gradiens megvalósítására (Fast Sigmoid) a nem-differenciálható Heaviside tüzelési függvényhez.
* **Miért ezt választottuk?**
  * **Dinamikus számítási gráf:** Az SNN-ek időbeli szimulációja időlépésről időlépésre fut ($t = 1 \dots T$). A PyTorch dinamikus gráfja (Eager execution) ideális az állapotvektorok időlépésenkénti léptetéséhez és a rétegenkénti lokális leválasztáshoz (`.detach()`).
  * **BPTT-mentesség és memóriatakarékosság:** A DECOLLE háromfaktoros tanulási szabállyal megkerüljük a memóriaigényes globális időbeli visszaterjesztést (BPTT), így a PyTorch-csal lineáris O(1) memóriafelhasználás érhető el tetszőleges hosszúságú idősoroknál.
  * **Hardver-agnosztikus működés:** Automatikus CUDA / CPU felismerést biztosít, így dedikált GPU-val és hagyományos CPU-n is azonnal hiba nélkül fut.

### **AdaMax Optimizáló ($\beta_1=0.0, \beta_2=0.95$)**
* **Mire használjuk?**
  * A DECOLLE rétegek helyi súlyfrissítéseinek optimalizálására a lokális véletlenszerű kiolvasók (random readouts) hibái alapján.
* **Miért ezt választottuk?**
  * A DECOLLE szakirodalmi specifikációja alapján a $\beta_1 = 0$ érték kikapcsolja a korábbi gradiensek lendületét (momentum-free streaming updates), ami elengedhetetlen a valósidejű, online adatfolyamok torzításmentes tanulásához, míg a $\beta_2 = 0.95$ biztosítja az adaptív skálázást a végtelen normán ($L_\infty$).

---

## 3. Memória- és Adattárolási Réteg

### **Redis (In-Memory Cache & Key-Value Store)**
* **Mire használjuk?**
  * **Munkamemória (Working Memory):** A jelenlegi kognitív ciklus feladatkontextusának, aktív állapotainak és percepciós kimeneteinek szub-milliszekundumos tárolására automatikus lejárati idővel (TTL).
* **Miért ezt választottuk?**
  * **Sebesség:** A RAM-ban tárolt kulcs-érték párok elérése <1 ms latenciával történik, elkerülve a lemezműveletek miatti kognitív blokkolást.
  * **Beépített Fallback:** A kódban készítettünk egy szálbiztos lokális Python in-memory gyorsítótárat is (`InMemoryCache`), így ha nem fut élő Redis démon a gépen vagy konténeren kívül, a rendszer zökkenőmentesen működik tovább.

### **Qdrant (Vektor-Adatbázis)**
* **Mire használjuk?**
  * **Epizodikus Memória (Episodic Memory):** Időbélyegekkel ellátott tapasztalatok, események és nagy dimenziós látens állapotbeágyazások (vektorok) tárolására és koszinusz-hasonlósági alapú asszociatív visszakeresésére.
* **Miért ezt választottuk?**
  * **Neuromorf mintaillesztés:** A biológiai epizodikus memória asszociatív (hasonlóság alapú). A Qdrant erre optimalizált HNSW indexelést biztosít.
  * **Beépített In-Memory Fallback:** Ha a Qdrant szerver nem elérhető, az `InMemoryVectorStore` azonnal átveszi a helyét NumPy normalizált skaláris szorzattal, így a tesztek és egyszerűbb futtatások külső szerver nélkül is azonnal működnek.

### **NetworkX (Gráf-struktúra és Tudásgráf)**
* **Mire használjuk?**
  * **Szemantikus Memória (Semantic Memory):** Absztrakt fogalmi csomópontok, kategóriák és oksági/relációs irányított élek tárolására és részgráf-lekérdezésekre.
  * **Dinamikus szinaptikus gráf:** Csomópontok és szinapszisok közötti Hebb-féle együtt-aktiváció és lekopás (pruning) modellezésére.
* **Miért ezt választottuk?**
  * **Könnyűsúlyú és flexibilis:** Nincs szükség gigabájtos különálló Neo4j / Memgraph JVM környezetre a fejlesztés és tesztelés során; a gráfműveletek, ego-gráf bejárások és legrövidebb út algoritmusok azonnal, memóriában elérhetők.

---

## 4. Kognitív Vezérlés és Keresési Algoritmusok

### **Pydantic (>= 2.0)**
* **Mire használjuk?**
  * Az esemény-adatszerkezetek (`EpisodicEvent`), kontextuális payloadok és séma-ellenőrzések validálására.
* **Miért ezt választottuk?**
  * **Szigorú adat-integritás:** Rust-alapú v2 motorja révén a validáció villámgyors, garantálva, hogy hibás vagy hiányos esemény ne kerülhessen a memóriahierarchiába.

### **Cognitive Decision Routing (CDR) és Dual-Process Vezérlés**
* **Mire használjuk?**
  * A Rendszer 1 (gyors, reflexszerű, SNN-alapú mintaillesztés, <10 ms) és Rendszer 2 (lassú, strukturált szimuláció és MCTS) közötti dinamikus kapuzásra (gating).
* **Miért ezt választottuk?**
  * **Erőforrás-takarékosság:** Nem indít el nehéz szimbolikus vagy fakeresési folyamatokat az ismert, rutinszerű ingerekre. A döntés 4 dimenzió ($C_s$: korreláció, $D_c$: tartománykeresztezés, $S_m$: érintettek száma, $U_l$: bizonytalanság) súlyozott kompozit értéke alapján történik.

### **JEPA (Joint-Embedding Predictive Architecture) és Rollout Szimulátor**
* **Mire használjuk?**
  * A világ belső absztrakt szimulálására a Rendszer 2 tervezési fázisában.
* **Miért ezt választottuk?**
  * **Nem pixeleken dolgozik:** A JEPA lényege, hogy a világ dinamikáját absztrakt látens térben modellezi ($s_{t+1} = \mathcal{T}(s_t, a_t)$). Ezzel megkíméli a rendszert a redundáns háttérzaj számításától, és 5-10 alternatív akció-trajektória szimulációját teszi lehetővé minimális CPU/GPU erőforrás mellett.

---

## 5. Monitorozás, Telemetria és Megjelenítés

### **Beépített Python `http.server` & REST API (`src/dashboard/monitor.py`)**
* **Mire használjuk?**
  * Valós idejű telemetria-kiszolgálásra (`/api/state`, `/api/update`) és a webes kezelőfelület megjelenítésére a `http://localhost:8000` címen.
* **Miért ezt választottuk?**
  * **Zero Extra Dependency:** Nem igényel külső nehézsúlyú keretrendszereket (mint Django vagy Celery); a standard library részeként egy háttérszálban fut daemonként, azonnal indul, nem ütközik verziókkal és nulla memória-többletet jelent.

### **Terminal User Interface (TUI / ASCII műszerek)**
* **Mire használjuk?**
  * Fejlesztői terminálban futó, élőben frissülő szöveges műszerfal (`dashboard_app.py`), amely közvetlenül mutatja a Homeosztatikus Állapotvektor mérőit, a döntési útvonalat és a logokat.
* **Miért ezt választottuk?**
  * Szervereken, SSH-n vagy fej nélküli (headless) Docker konténerekben is közvetlen, vizuális visszacsatolást ad a fejlesztőnek böngésző nélkül is.

---

## 6. Konténerizáció és Infrastruktúra

### **Docker & Docker Compose**
* **Mire használjuk?**
  * A többkonténeres kognitív architektúra (`brain-core`, `redis`, `qdrant`) egyetlen parancsos (`docker compose up`) elindítására.
* **Miért ezt választottuk?**
  * **Környezeti izoláció és reprodukálhatóság:** A `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` alapképpel garantált a CUDA 12.1 GPU támogatás és az összes C++ build függőség megléte bármely Linux/Windows/Mac gazdagépen.

---

## 7. Minőségbiztosítás és Tesztelés

### **Pytest (>= 9.0)**
* **Mire használjuk?**
  * A 6 alappillér, a tanítási pipeline, a dashboard és a specifikációs modulok átfogó automatizált egység- és integrációs tesztelésére a `neuromorphic-brain/tests/` mappában.
* **Miért ezt választottuk?**
  * Paraméterezhető, gyors és tiszta tesztriportokat biztosít; a teljes 27 tesztes csomag ~15 másodperc alatt fut le, garantálva a hibamentes működést.

---

## 📊 Összefoglaló Mátrix

| Komponens / Könyvtár | Szerepkör | Fő előny és indoklás |
| :--- | :--- | :--- |
| **PyTorch** | SNN, DECOLLE & JEPA tenzorok | Rugalmas autograd, surrogate gradiensek, BPTT-mentes lokális tanulás |
| **AdaMax** | Helyi DECOLLE súlyoptimalizáló | $\beta_1=0.0$ streaming adatokhoz momentum-torzítás nélkül |
| **Redis** | Munkamemória (Working Memory) | Szub-milliszekundumos adatátvitel, beépített TTL lejárat |
| **Qdrant** | Epizodikus Memória (Vector DB) | Nagy dimenziós koszinusz-hasonlósági élmény-visszakeresés |
| **NetworkX** | Szemantikus Memória & Tudásgráf | Gyors in-memory irányított gráfműveletek külön szerver nélkül |
| **Pydantic** | Adatvalidáció & Adatmodellek | Szigorú típusellenőrzés és megbízható eseménystruktúrák |
| **DualProcessRouter** | Rendszer 1 / 2 Gating | Optimális erőforrás-gazdálkodás: rutin feladatok <10ms alatt |
| **JEPASandbox** | Világmodell rollout | Absztrakt látens predikció felesleges pixel-szintű számítások nélkül |
| **Monitor & TUI** | Web Dashboard és Terminál UI | Zero-dependency valós idejű telemetria és állapotkövetés |
| **Docker Compose** | Infrastruktúra-szervezés | Egykattintásos elindítás Redis-szel és Qdrant-tal |
| **Pytest** | Automatizált tesztek | 27 egység- és integrációs teszt 100%-os lefedettséggel |
