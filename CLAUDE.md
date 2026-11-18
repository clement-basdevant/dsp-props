# dsp-props

Couche de property-based testing pour le traitement du signal, construite sur Hypothesis.
Au lieu de comparer à une sortie de référence, on vérifie des relations métamorphiques qui
doivent tenir pour toute implémentation correcte, avec une tolérance dérivée d'un modèle
d'erreur, et on rapporte le plus petit contre-exemple.

Auteur : Clément Basdevant (Nicolas Boganski, retiré des auteurs pour l'instant, est cité dans les remerciements). Papier JOSS en préparation (`paper/paper.md`, `paper/paper.bib`).
Version 0.1.4 publiée sur PyPI (procédure : `RELEASING.md`). Plan de développement :
`docs/schema-directeur.md`. Tickets de bugs à poster en amont : `papers/tickets-a-poster.md`.

## Les trois briques

1. **Catalogue de relations** : linéarité, théorèmes de décalage et de convolution, Parseval,
   symétrie hermitienne (DFT réelle : N//2+1 bins, DC et Nyquist réels), inversibilité,
   invariance temporelle, causalité, cohérence avec une fonction de transfert déclarée,
   invariance temporelle périodique d'un rééchantillonneur L/M, gain borné sur N <= max_n
   (pas une preuve de stabilité BIBO), invariance au découpage en blocs d'un processeur en
   streaming (découpages tirés par `block_splits` : coupures arbitraires jusqu'à max(N, 16)
   blocs, vides et d'un échantillon compris, blocs d'un échantillon, blocs de taille fixe),
   cohérence float32/float64 (`float32=True` sur une déclaration float64 : sortie float32
   proche de la sortie float64 à ε float32, entrées réelles seulement ; action 2.3).
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
   `dp.calibrate` (chemin recommandé pour une tolérance serrée, avec `base=None`) mesure une implémentation
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

Plugin pytest (`dsp_props/pytest_plugin.py`, point d'entrée `pytest11`) : un opérateur déclaré
dans un module de test (ou listé dans `dsp_props_operators`) devient un test par relation
(`test_x.py::my_fft::parseval`). Profils de coût : `fast` (30 entrées), `default` (200),
`nightly` (2000), par `--dsp-props-profile`, `--profile`, `profile=` ou `DSP_PROPS_PROFILE` ;
un `max_examples` explicite l'emporte. Base d'exemples Hypothesis `.hypothesis/examples`
rejouée en premier (inactive avec `seed=` ou `derandomize`). Tests de non-régression générés
(`dsp_props/regression.py`) : `--dsp-props-regressions [DIR]`, `dsp-props check --regressions DIR`
ou `dp.write_regression_tests` écrivent `test_regression_<op>.py`, un test pytest simple par
contre-exemple minimal, rejoué exactement sans Hypothesis. Doc : `docs/pytest.md`.

Calibrations versionnées (`dsp_props/tolerances.py`, action 2.2) : `dsp-props calibrate CIBLES
--tolerances tolerances.json` mesure chaque opérateur contre une référence étendue (DFT étendue
pour `dft`/`rdft`, `longdouble_lfilter` pour un filtre avec `transfer=`) et stocke les rapports
par longueur avec les versions des bibliothèques (`dp.ToleranceFile`) ; `dp.verify(tolerances=)`,
`dsp-props check --tolerances`, `--dsp-props-tolerances` et `DSP_PROPS_TOLERANCES` utilisent
alors la calibration au lieu du modèle par défaut (une tolérance déclarée l'emporte).
`--compare` : régression de précision entre deux versions (erreur > 2 × l'enregistrée, mêmes
signaux sondes). Doc : `docs/tolerances.md`.

## Commandes

```bash
python -m pip install -e ".[dev]"      # installation de développement
python -m pytest -q                    # suite de tests interne (~5 min ; profil fast en CI : DSP_PROPS_PROFILE=fast)
ruff check . && mypy                   # lint et types (job CI lint)
dsp-props check module_or_file.py      # découvre les opérateurs déclarés, code de sortie ≠ 0 si échec
python mutants/run_mutants.py          # suite de mutation (31 mutants, 6 familles)
python benchmarks/precision_mutants.py # mutants de précision : plus petite perte détectée (~10 min)
python benchmarks/precision_oracles.py # mêmes mutants contre 4 oracles (A9, ~1 h)
python benchmarks/calibrated_tolerance.py # tolérance calibrée : fausses alarmes, 5 filtres, 4 rapports (~1 h)
python benchmarks/campaign.py          # campagne sur 46 cibles réelles
python benchmarks/error_profile.py     # profil d'erreur FFT, N = 2..1200
dsp-props calibrate mod.py --tolerances tolerances.json [--compare]  # calibration versionnée
python benchmarks/make_results.py      # chiffres cités (results.md/.tex, marqueurs <!-- r:clé -->) ; --check en CI
```

La CI (`.github/workflows/ci.yml`) lance pytest (profil fast, couverture, base d'exemples en
cache), la suite de mutation, la CLI sur `examples/filters.py` et la construction du paquet,
sous Python 3.10, 3.12 et 3.13 ; un job lint (ruff, mypy, `make_results.py --check`). Chaque
nuit (2 h 17 UTC) : suite au profil default et chasse au profil nightly sur les exemples.

## Règles du projet

- **Ne jamais élargir une tolérance pour faire passer un test.** Si une relation échoue,
  c'est soit un bug de l'opérateur, soit un modèle d'erreur faux à corriger en le justifiant.
  `dp.calibrate` refuse volontairement les erreurs implausibles plutôt que de les absorber.
- **Chaque chiffre du papier vient d'un script du dépôt** (les trois ci-dessus). Après toute
  modification des relations, des générateurs ou des tolérances, relancer les scripts et
  mettre à jour le papier si les chiffres changent. Dans le README, `CLAUDE.md` et `docs/`, un chiffre issu d'un
  benchmark s'écrit `<!-- r:clé -->valeur<!-- /r -->` (clés dans `benchmarks/results.md`) ;
  `python benchmarks/make_results.py` les réécrit depuis les JSON, la CI vérifie (`--check`).
- Un contre-exemple n'est utile que s'il est petit : surveiller la qualité du shrinking
  (actuellement médiane de 2 échantillons).
- L'API déclarative doit rester légère : un décorateur par opérateur.

## État actuel (octobre 2026)

- Mutants : <!-- r:mutants.killed -->31<!-- /r -->/<!-- r:mutants.total -->31<!-- /r --> détectés avec <!-- r:mutants.examples -->200<!-- /r --> entrées par relation. Ils sont écrits à la main, dans le
  vocabulaire des relations : le score mesure que les relations attrapent les fautes pour
  lesquelles elles ont été conçues. `run_mutants.py` ne fixe pas de graine ; avec
  `dp.verify(seed=0)`, 28/31 échouent dans les 3 premières entrées et 26/30 contre-exemples
  ont au plus 3 échantillons (médiane 2). Angle mort connu : un filtre faux mais toujours LTI
  et causal, détecté dès qu'on déclare sa fonction de transfert.
- Mutants de précision (action M-1, `mutants/precision_mutants.py`) : mêmes propriétés, moins
  de bits significatifs (0 à 24 bits perdus). Avec la tolérance par défaut, 3 graines : FFT
  attrapée à partir d'environ <!-- r:precision.fft_twiddles.default -->90 ε<!-- /r --> (<!-- r:precision.fft_twiddles.passes_up_to -->21 ε<!-- /r --> passe, twiddles récursifs à <!-- r:precision.realistic.fft_radix2_recursive_twiddles.loss -->19 ε<!-- /r --> passent), streaming
  à partir de <!-- r:precision.streaming_state.default -->7.1e3 ε<!-- /r -->, rééchantillonneur (arithmétique) de <!-- r:precision.resampler_arithmetic.default -->1.1e5 ε<!-- /r -->, IIR de <!-- r:precision.iir_coefficients.default -->7.1e5 ε<!-- /r --> (coefficients) à
  <!-- r:precision.iir_arithmetic.default -->3.0e6 ε<!-- /r --> (arithmétique). Coefficients arrondis d'un rééchantillonneur : jamais attrapés par les relations (ni par la cohérence float32/float64, l'arrondi étant le même dans les deux précisions) ; `dsp-props calibrate --compare` contre le rééchantillonneur idéal les signale dès 4 bits perdus (test).
  Détail dans `docs/blind-spots.md`.
- A9 (`benchmarks/precision_oracles.py`) : tolérance fixe 1e-7 ne détecte aucun de ces
  mutants ; tolérance calibrée sur l'implémentation de confiance, sans plancher analytique
  (`CalibratedTolerance(..., base=FFTTolerance(c=0))`) : FFT dès <!-- r:oracles.fft_twiddles.calibrated -->21 ε<!-- /r -->, streaming <!-- r:oracles.streaming_state.calibrated -->28 ε<!-- /r -->, filtres
  <!-- r:oracles.resampler_arithmetic.calibrated -->430 ε<!-- /r --> à <!-- r:oracles.iir_arithmetic.calibrated -->2.9e3 ε<!-- /r -->, sans fausse alarme (3 graines). `dp.calibrate` garde par défaut le plancher
  20 ε log2 N (il élargit seulement) ; `base=None` le retire. `assert_allclose` différentiel :
  inégal. `benchmarks/calibrated_tolerance.py` : tolérance calibrée sans plancher sur
  l'implémentation de confiance, aucune fausse alarme sur <!-- r:calibrated.correct_without_alarm -->13<!-- /r --> autres implémentations correctes
  (scipy.fft, radix-2, DFT directe, sosfilt sur 5 filtres, polyphase sur 4 rapports, convolve) ;
  seule la FFT PyTorch (<!-- r:calibrated.torch_fft -->156 ε<!-- /r --> contre <!-- r:calibrated.numpy_fft -->2.7 ε<!-- /r --> pour NumPy) est signalée. Sur 5 filtres et 4
  rapports, calibré <!-- r:calibrated.gain_min -->7<!-- /r --> à <!-- r:calibrated.gain_max -->3400<!-- /r --> fois plus fin que le défaut : le modèle IIR dérivé est pessimiste.
  La calibration suppose une implémentation de confiance et une référence étendue.
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
  double-double ; écart de précision entre la tolérance par défaut (20 ε log2 N) et le refus de
  `dp.calibrate` (1e6 ε), où une FFT à facteurs de rotation récursifs (38 ε) passe.
- Résultats en amont, à leur juste taille :
  - `lfilter` sur bloc vide avec état initial : redécouverte de scipy/scipy#22571, déjà
    corrigé dans la version de développement ; le cas FIR se propage à la pré-emphase en
    streaming de librosa.
  - `sosfilt` sur bloc vide : nouveau (scipy/scipy#26406, avec `lfilter`), mais c'est une
    exception levée sur une entrée vide, pas une divergence silencieuse. Corrigé (avec le
    cas `lfilter`) par notre PR scipy/scipy#26415, fusionnée le 23/10/2026 par un
    mainteneur, prévue pour SciPy 2.0.0.
  - Pas encore de confirmation d'un mainteneur pour les trois points suivants.
  - `tf2sos` qui supprime les retards purs (scipy/scipy#26417) : comportement documenté,
    peut-être un problème de documentation.
  - `deemphasis` de torchaudio (pytorch/audio#4238) : écrêtage par défaut (`clamp=True`)
    documenté, bibliothèque en maintenance.
  - `filtfilt(method="gust")` dépend de l'écriture du filtre (scipy/scipy#26610) : des
    zéros en fin de `a` ou `b` gonflent l'ordre, ajoutent une impulsion aux bords (0,54 sur
    un exemple) et rendent le système des moindres carrés singulier. Reproduit de 1.15 à 1.18.1.
  - Prolongement (campagne 3, trié le 18/11, scipy/scipy#26755) : des pôles proches de zéro,
    sans être nuls, rendent ce système quasi singulier et `gust` perd des chiffres sur un
    problème bien conditionné ; la PR scipy#26624 (zéros exacts) ne le couvre pas. Analyse :
    `benchmarks/gust_near_zero_poles.py`.
  - SciPy applique une politique IA : tickets et PR rédigés par l'auteur, pas par une IA.
- FFT CPU de PyTorch : jusqu'à 195 ε pour certaines longueurs à grands facteurs premiers
  (N = 596), contre environ 4 ε pour NumPy/SciPy.
- Fausses alarmes : `benchmarks/validation.json` (relancé après #7 à #10) donne 2000
  exécutions sur du code correct, aucun échec. `python benchmarks/validation.py --only
  false-positives --seeds 1` refait la vérification en environ 2 min (100 exécutions).

## Pistes ouvertes (fiche de projet)

- CLI qui devine la catégorie d'une fonction non déclarée (signature, nom, docstring).
- Relations de multicadence avancées (identités nobles), stabilité BIBO, dénormaux
  (flush-to-zero).
- Code C embarqué en virgule fixe.
- Propriétés d'estimateurs : biais, variance contre la borne de Cramér-Rao.
