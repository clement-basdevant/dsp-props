# Campagne 3 du 18 novembre 2026 : relance avec le `block_splits` de la bibliothèque

Environnement : Python 3.13, SciPy 1.18.1, NumPy 2.5.3, Hypothesis 6.170.15, librosa 1.0.1,
PyTorch 2.15.0 (CPU), torchaudio 2.11.1.
Commande : `python benchmarks/campaign.py --examples 200` : 46 cibles, 39 passent.

Changement : `campaign.py` n'a plus sa propre copie de `block_splits` (au plus 8 coupures, donc
9 blocs). Il importe celle de la bibliothèque : coupures arbitraires jusqu'à max(N, 16) blocs,
blocs d'un échantillon, blocs de taille fixe.

## Six échecs déjà connus, un à trier

| Cible | Contre-exemple minimal | Ticket |
|---|---|---|
| `scipy.signal.lfilter` par blocs | x = [1], un bloc vide en tête | scipy#22571 / scipy#26406 (corrigé, SciPy 2.0.0) |
| `scipy.signal.sosfilt` par blocs | x = [1], un bloc vide en tête | scipy#26406 (corrigé, SciPy 2.0.0) |
| `lfilter` avec `zi` via `dp.stateful` | bloc vide | scipy#26406 |
| `librosa.effects.preemphasis` par blocs | x = [1], bloc vide | conséquence de scipy#26406 |
| `sosfilt(tf2sos(b, a))` contre `lfilter(b, a)` | b = [0, 1], a = [1, 0, 0] | scipy#26417 |
| `torchaudio` deemphasis(preemphasis(x)) | x = [1000] | pytorch/audio#4238 (écrêtage `clamp=True`) |

## À trier : `filtfilt(method="gust")` et renversement temporel

Contre-exemple : x = 97 échantillons à 1, b = [1], a d'ordre 6 dont quatre pôles sont groupés
autour de 0,5 (0,50008 ± 7,8e-5 j et 0,49992 ± 7,8e-5 j) et deux sont à 0,001. Erreur L2
2,1e-7 pour une tolérance de 3,5e-8 (c = 1e4, propre à cette cible), soit 9e-11 en relatif.
Ce n'est pas le cas de scipy#26610 (pas de zéros en fin de `a` ou `b`). Des pôles quasi
confondus rendent le filtre très mal conditionné ; l'écart vient probablement de là plutôt
que d'un bug, mais la tolérance de cette cible est une constante d'ingénierie, non dérivée.
On ne l'élargit pas : à trier (modèle d'erreur tenant compte du conditionnement, ou
restriction documentée des filtres tirés).

## Comparaison avec la campagne 2

Campagne 2 (14/10) : 40/46. La seule différence est cette cible `gust`, qui passait. Elle
n'utilise pas `block_splits` : la campagne ne fixe pas de graine et ce filtre n'avait pas été
tiré. Le nouveau générateur de découpages ne change pas les résultats : les deux cibles par
blocs échouaient déjà sur un bloc vide et retrouvent le même contre-exemple d'un échantillon.
