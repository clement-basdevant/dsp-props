# Campagne 3 du 18 novembre 2026 : relance avec le `block_splits` de la bibliothèque

Environnement : Python 3.13, SciPy 1.18.1, NumPy 2.5.3, Hypothesis 6.170.15, librosa 1.0.1,
PyTorch 2.15.0 (CPU), torchaudio 2.11.1.
Commande : `python benchmarks/campaign.py --examples 200` : 46 cibles, 39 passent.

Changement : `campaign.py` n'a plus sa propre copie de `block_splits` (au plus 8 coupures, donc
9 blocs). Il importe celle de la bibliothèque : coupures arbitraires jusqu'à max(N, 16) blocs,
blocs d'un échantillon, blocs de taille fixe.

## Six échecs déjà connus, un nouveau (trié plus bas)

| Cible | Contre-exemple minimal | Ticket |
|---|---|---|
| `scipy.signal.lfilter` par blocs | x = [1], un bloc vide en tête | scipy#22571 / scipy#26406 (corrigé, SciPy 2.0.0) |
| `scipy.signal.sosfilt` par blocs | x = [1], un bloc vide en tête | scipy#26406 (corrigé, SciPy 2.0.0) |
| `lfilter` avec `zi` via `dp.stateful` | bloc vide | scipy#26406 |
| `librosa.effects.preemphasis` par blocs | x = [1], bloc vide | conséquence de scipy#26406 |
| `sosfilt(tf2sos(b, a))` contre `lfilter(b, a)` | b = [0, 1], a = [1, 0, 0] | scipy#26417 |
| `torchaudio` deemphasis(preemphasis(x)) | x = [1000] | pytorch/audio#4238 (écrêtage `clamp=True`) |

## Trié le 18/11 : `filtfilt(method="gust")` perd la précision quand des pôles sont proches de zéro

Contre-exemple : x = 97 échantillons à 1, b = [1], a d'ordre 6 dont quatre pôles sont groupés
autour de 0,5 (0,50008 ± 7,8e-5 j et 0,49992 ± 7,8e-5 j) et deux sont à 0,001. Erreur L2
2,1e-7 pour une tolérance de 3,5e-8 (c = 1e4, propre à cette cible), soit 9e-11 en relatif.

Verdict : **défaut numérique de SciPy, pas un problème mal conditionné.** La tolérance ne
change pas. Détail et chiffres : `python benchmarks/gust_near_zero_poles.py`
(`gust_near_zero_poles.json`), qui rejoue l'algorithme de SciPy en 80 chiffres (mpmath) sur
les mêmes entrées float64 :

- Le problème est bien conditionné : nombre de conditionnement relatif de x ↦ y d'environ 1,
  sensibilité aux coefficients de `a` d'au plus environ 100. Une implémentation stable devrait
  donc se tromper d'environ 100 ε au plus.
- SciPy se trompe de 1,9e7 ε sur le cas de la campagne (4e-9 en relatif). L'écart au
  renversement temporel (7e4 ε) n'est que la partie visible de cette erreur.
- La cause n'est pas le groupe de pôles à 0,5 (seul : 24 ε) mais **la paire de pôles à
  0,001**. Avec a = poly([0,5, p, p]), l'erreur croît quand p diminue : 8,8e3 ε pour p = 1e-2,
  3,2e7 ε pour p = 1e-4, 7,3e9 ε (1,6e-6 en relatif) pour p = 1e-6 ; à p = 1e-7, 5e-3 en
  relatif. Avec p = 0 exactement, l'erreur redevient 0,3 ε.
- Mécanisme : les états initiaux de Gustafsson sont la solution d'un moindre carré dont la
  matrice M devient quasi singulière (cond(M) ≈ 5e12 pour p = 1e-6) ; les arrondis du second
  membre (écart entre filtrage avant-arrière et arrière-avant) sont amplifiés par 1/σ_min.
  Tronquer la SVD ne corrige pas : la direction faible porte une partie réelle de la solution.
- C'est le prolongement continu de scipy#26610 (zéros exacts en fin de `a`) : la PR
  scipy#26624 retire les zéros exacts et ne couvre pas les coefficients minuscules non nuls.
  Aucun ticket SciPy existant ne le signale (recherche du 18/11).

Suite : ticket SciPy posté le 18/11 par Clément, scipy/scipy#26755. La cible de campagne reste telle quelle : elle
détecte un vrai défaut, et doit passer quand SciPy l'aura corrigé.

## Comparaison avec la campagne 2

Campagne 2 (14/10) : 40/46. La seule différence est cette cible `gust`, qui passait. Elle
n'utilise pas `block_splits` : la campagne ne fixe pas de graine et ce filtre n'avait pas été
tiré. Le nouveau générateur de découpages ne change pas les résultats : les deux cibles par
blocs échouaient déjà sur un bloc vide et retrouvent le même contre-exemple d'un échantillon.
