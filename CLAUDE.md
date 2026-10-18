# dsp-props

Couche de property-based testing pour le traitement du signal, construite sur Hypothesis.
Au lieu de comparer à une sortie de référence, on vérifie des relations métamorphiques qui
doivent tenir pour toute implémentation correcte, avec une tolérance dérivée d'un modèle
d'erreur, et on rapporte le plus petit contre-exemple.

Auteur : Clément Basdevant (Nicolas Boganski, retiré des auteurs pour l'instant, est cité dans les remerciements). Papier JOSS en préparation (`paper/paper.md`, `paper/paper.bib`).
Version 0.1.2 publiée sur PyPI (procédure : `RELEASING.md`). Plan de développement :
`docs/schema-directeur.md`. Tickets de bugs à poster en amont : `papers/tickets-a-poster.md`.

## Les trois briques

1. **Catalogue de relations** : linéarité, théorèmes de décalage et de convolution, Parseval,
   symétrie hermitienne (DFT réelle : N//2+1 bins, DC et Nyquist réels), inversibilité,
   invariance temporelle, causalité, cohérence avec une fonction de transfert déclarée,
   invariance temporelle périodique d'un rééchantillonneur L/M, gain borné sur N <= max_n
   (pas une preuve de stabilité BIBO), invariance au découpage en blocs d'un processeur en
   streaming (découpages tirés par `block_splits` : coupures arbitraires jusqu'à max(N, 16)
   blocs, vides et d'un échantillon compris, blocs d'un échantillon, blocs de taille fixe).
2. **Générateurs de signaux** (stratégies Hypothesis) : impulsions, constantes, sinusoïdes
   hors grille et quasi-Nyquist, paires de tons proches, signaux parcimonieux, bruit blanc ;
   longueurs pathologiques (1, 2, premiers, 2^k ± 1 : environ 15 % des longueurs tirées sur
   200 exemples) ; amplitudes de 1e-100 à 1e100 (1, 1e-3, 1e3 en float32) ;
   découpages en blocs avec blocs vides et d'un échantillon.
3. **Moteur de tolérance** : FFT en c · ε · log2 N, forme de la borne pire cas de Higham
   (2002, théorème 24.2 ; Schatzman 1996 montre une erreur quadratique moyenne en
   O(ε √log N)), mais c = 20 (transformées) et c = 1000 (filtres sans fonction de transfert,
   rééchantillonneurs, streaming) sont des constantes d'ingénierie, pas des bornes dérivées.
   Seul le modèle des filtres récursifs (`core/iir.py`) est dérivé, au premier ordre et pour
   la forme directe I ; il croît avec l'ordre et les normes ℓ1 des réponses impulsionnelles.
   `dp.calibrate` (chemin recommandé pour une tolérance serrée) mesure une implémentation
   longueur par longueur contre `extended_dft` : FFT `long double` là où elle est étendue,
   DFT double-double sinon.

## API

```python
import dsp_props as dp

@dp.transform(inverse=np.fft.ifft, parseval=True, real_input_hermitian=True)
def my_fft(x): return np.fft.fft(x)

dp.verify(my_fft)   # lève PropertyViolation avec un contre-exemple minimal
```

Déclarations disponibles : transformées (complexe, entrée réelle, linéaire générique comme
la DCT), filtres LTI (fonction de transfert et borne de gain optionnelles), rééchantillonneurs
rationnels, processeurs en streaming ou à état. Adaptateurs NumPy, SciPy, PyTorch, et C via
ctypes.

## Commandes

```bash
python -m pip install -e ".[dev]"      # installation de développement
python -m pytest -q                    # suite de tests interne (123 tests, ~5 min)
dsp-props check module_or_file.py      # découvre les opérateurs déclarés, code de sortie ≠ 0 si échec
python mutants/run_mutants.py          # suite de mutation (31 mutants, 6 familles)
python benchmarks/campaign.py          # campagne sur 46 cibles réelles
python benchmarks/error_profile.py     # profil d'erreur FFT, N = 2..1200
```

La CI (`.github/workflows/ci.yml`) lance pytest, la suite de mutation, la CLI sur
`examples/filters.py` et la construction du paquet, sous Python 3.10, 3.12 et 3.13.

## Règles du projet

- **Ne jamais élargir une tolérance pour faire passer un test.** Si une relation échoue,
  c'est soit un bug de l'opérateur, soit un modèle d'erreur faux à corriger en le justifiant.
  `dp.calibrate` refuse volontairement les erreurs implausibles plutôt que de les absorber.
- **Chaque chiffre du papier vient d'un script du dépôt** (les trois ci-dessus). Après toute
  modification des relations, des générateurs ou des tolérances, relancer les scripts et
  mettre à jour le papier si les chiffres changent.
- Un contre-exemple n'est utile que s'il est petit : surveiller la qualité du shrinking
  (actuellement médiane de 2 échantillons).
- L'API déclarative doit rester légère : un décorateur par opérateur.

## État actuel (octobre 2026)

- Mutants : 31/31 détectés avec 200 entrées par relation. Ils sont écrits à la main, dans le
  vocabulaire des relations : le score mesure que les relations attrapent les fautes pour
  lesquelles elles ont été conçues. `run_mutants.py` ne fixe pas de graine ; avec
  `dp.verify(seed=0)`, 28/31 échouent dans les 3 premières entrées et 26/30 contre-exemples
  ont au plus 3 échantillons (médiane 2). Angle mort connu : un filtre faux mais toujours LTI
  et causal, détecté dès qu'on déclare sa fonction de transfert.
- Changements récents (PR #7 à #10, fusionnées) :
  - #8 : la relation « transfer function » borne la queue tronquée de la réponse
    impulsionnelle (`tail_l1_bound`, solution de Lyapunov) au lieu de supposer 1e-17 ·
    ||h||_1 ; fin de la fausse alarme sur un `lfilter` correct à pôle proche du cercle.
  - #10 : la précision attendue vient de la déclaration (`precision=`, float64 par défaut),
    plus du dtype de sortie ; une sortie moins précise que l'entrée est un échec.
  - #9 : `dp.calibrate` utilise `extended_dft` (FFT `long double` si elle est vraiment
    étendue, sinon DFT double-double en O(N²)) ; job CI macOS arm64 sur ce chemin.
  - #7 : `block_splits` mélange coupures arbitraires (jusqu'à max(N, 16) blocs), blocs d'un
    échantillon et blocs de taille fixe (avant : au plus 8 blocs).
- Limites restantes (détail dans `docs/blind-spots.md`) : fenêtre de réponse impulsionnelle
  plafonnée à 2^16 ; une précision par déclaration ; `longdouble_lfilter` sans repli
  double-double ; `benchmarks/campaign.py` garde sa propre copie de `block_splits` plafonnée
  à 9 blocs ; écart de précision entre la tolérance par défaut (20 ε log2 N) et le refus de
  `dp.calibrate` (1e6 ε), où une FFT à facteurs de rotation récursifs (38 ε) passe.
- Résultats en amont, à leur juste taille (aucune confirmation d'un mainteneur à ce jour) :
  - `lfilter` sur bloc vide avec état initial : redécouverte de scipy/scipy#22571, déjà
    corrigé dans la version de développement ; le cas FIR se propage à la pré-emphase en
    streaming de librosa.
  - `sosfilt` sur bloc vide : nouveau (scipy/scipy#26406, avec `lfilter`), mais c'est une
    exception levée sur une entrée vide, pas une divergence silencieuse.
  - `tf2sos` qui supprime les retards purs (scipy/scipy#26417) : comportement documenté,
    peut-être un problème de documentation.
  - `deemphasis` de torchaudio (pytorch/audio#4238) : écrêtage par défaut (`clamp=True`)
    documenté, bibliothèque en maintenance.
  - SciPy applique une politique IA : tickets et PR rédigés par l'auteur, pas par une IA.
- FFT CPU de PyTorch : jusqu'à 195 ε pour certaines longueurs à grands facteurs premiers
  (N = 596), contre environ 4 ε pour NumPy/SciPy.
- `benchmarks/validation.py --only false-positives` est cassé depuis #10 : sa sonde
  `_instrumented_assert_close` n'accepte pas l'argument `precision=` (TypeError sur 62 des
  100 lignes avec `--seeds 1`). Les 2000 exécutions sans fausse alarme de
  `benchmarks/validation.json` datent d'avant #7 à #10.

## Pistes ouvertes (fiche de projet)

- CLI qui devine la catégorie d'une fonction non déclarée (signature, nom, docstring).
- Relations de multicadence avancées (identités nobles), stabilité BIBO, cohérence
  float32/float64, dénormaux.
- Génération automatique de tests de non-régression à partir des contre-exemples.
- Code C embarqué en virgule fixe.
- Propriétés d'estimateurs : biais, variance contre la borne de Cramér-Rao.
