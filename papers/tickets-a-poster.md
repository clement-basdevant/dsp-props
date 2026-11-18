# Tickets à poster

Brouillons prêts, à poster depuis le compte GitHub du projet. Une fois posté, remplacer « à poster » par le lien.

| # | Où | Statut |
| --- | --- | --- |
| 0 | scipy/scipy : lfilter/sosfilt sur bloc vide | posté : [scipy#26406](https://github.com/scipy/scipy/issues/26406), corrigé par [scipy#26415](https://github.com/scipy/scipy/pull/26415) (fusionnée le 23/10/2026, SciPy 2.0.0) |
| 1 | pytorch/audio : deemphasis écrête à [-1, 1] | posté : [pytorch/audio#4238](https://github.com/pytorch/audio/issues/4238) |
| 2 | scipy/scipy : tf2sos supprime les retards purs | posté : [scipy#26417](https://github.com/scipy/scipy/issues/26417) |
| 3 | commentaire sur scipy#26406 : impact librosa | posté le 15/10/2026 sur [scipy#26406](https://github.com/scipy/scipy/issues/26406) |
| 4 | scipy/scipy : filtfilt(method="gust") et zéros en fin de `a` ou `b` | posté le 04/11/2026 : [scipy#26610](https://github.com/scipy/scipy/issues/26610), correctif proposé dans [scipy#26624](https://github.com/scipy/scipy/pull/26624) (ouverte le 05/11/2026, en attente de revue) |
| 5 | scipy/scipy : filtfilt(method="gust") imprécis avec des pôles proches de zéro | posté le 18/11/2026 : [scipy#26755](https://github.com/scipy/scipy/issues/26755) |

---

## 1. pytorch/audio

Lien : https://github.com/pytorch/audio/issues/new (modèle Bug report)

**Titre :** deemphasis silently clamps output to [-1, 1], so it is not the inverse of preemphasis

### 🐛 Describe the bug

`torchaudio.functional.deemphasis` is documented as the "Inverse of `preemphasis`", but it calls `lfilter` with the default `clamp=True`, so its output is silently clamped to [-1, 1]. As a result `deemphasis(preemphasis(x))` does not return `x` as soon as `|x| > 1`, and the clamping is not mentioned in the documentation.

```python
import torch
import torchaudio.functional as F

x = torch.tensor([1000.0, 500.0], dtype=torch.float64)
print(F.deemphasis(F.preemphasis(x, 0.0), 0.0))
# tensor([1., 1.], dtype=torch.float64)   expected: tensor([1000., 500.])

x = torch.tensor([0.9, 0.9, 0.9], dtype=torch.float64)
print(F.deemphasis(x, 0.97))
# tensor([0.9000, 1.0000, 1.0000]): values above 1 are clamped although the input is a valid waveform
```

Suggested fix: pass `clamp=False` to `lfilter` in `deemphasis` (or expose a `clamp` argument and document it).

### Versions

torchaudio 2.11.0, PyTorch 2.14.1, Python 3.13, Linux x86_64 (CPU).

Found with dsp-props (https://github.com/clement-basdevant/dsp-props), a property-based testing layer for DSP code, through the relation `deemphasis(preemphasis(x)) == x`.

Related: #3730 (same silent clamping in `transforms.Vol`) and #596 (which made `clamp` optional in `lfilter`, default kept to True).

AI disclosure: the reproducer above was found with dsp-props and the text of this report was drafted with an AI assistant (Claude); I ran the code and checked the outputs myself on the versions below.

---

## 2. scipy/scipy

**Le texte doit être écrit par toi** : la politique IA de SciPy (https://docs.scipy.org/doc/scipy/dev/conduct/ai_policy.html) interdit d'utiliser une IA « pour parler à ta place » dans une issue, sauf pour traduire ou corriger la langue. Tu peux l'écrire en français et me demander la traduction. Pas de déclaration IA exigée pour une issue.

Lien : https://github.com/scipy/scipy/issues/new/choose (Bug report)

Faits vérifiés le 15/10/2026 sur SciPy 1.18.1, NumPy 2.5.4, Python 3.13 :
- Pour un filtre avec plus de pôles que de zéros, par exemple le retard pur `y[n] = x[n-2]` (`b = [0, 0, 1]`, `a = [1, 0, 0]`), `tf2sos` renvoie la section identité `[[1, 0, 0, 1, 0, 0]]`.
- Résultat : `lfilter(b, a, np.arange(5.))` donne `[0, 0, 0, 1, 2]` et `sosfilt(tf2sos(b, a), np.arange(5.))` donne `[0, 1, 2, 3, 4]`. Le retard est perdu, le module de la réponse en fréquence est conservé.
- Même chose avec `b = [0, 1, 0.5]`, `a = [1, -0.5]` : `[0, 0, 1, 3, 5.5]` contre `[0, 1, 3, 5.5, 8.25]`.
- Pas d'erreur. Pour `b = [0, 0, 1]`, SciPy émet seulement un avertissement `BadCoefficients` (« Badly conditioned filter coefficients (numerator) »), qui ne parle pas du retard.
- La doc de `zpk2sos` dit qu'elle ajoute « poles or zeros to the origin as necessary to obtain the same number of poles and zeros ». N'ajouter que des zéros à l'origine multiplie H(z) par z^k.
- Le cas a déjà été signalé dans #13375 (`tf2sos([1], [1, 0])`) puis retiré par son auteur sans réponse des mainteneurs. Ce n'est pas un doublon ouvert, mais il faut le citer.
- Aucun autre ticket ou PR ouvert ne couvre ce cas (recherche du 15/10/2026).

Code pour te reproduire le problème (à lancer toi-même) :

```python
import numpy as np
import scipy.signal as ss

b, a = [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]   # y[n] = x[n-2]
print(ss.tf2sos(b, a))
x = np.arange(5.0)
print(ss.lfilter(b, a, x))
print(ss.sosfilt(ss.tf2sos(b, a), x))
```

---

## 3. Commentaire sur scipy#26406

Lien : https://github.com/scipy/scipy/issues/26406

Additional impact downstream: `librosa.effects.preemphasis` (librosa 1.0.0) uses `scipy.signal.lfilter` and inherits the FIR case, so streaming pre-emphasis fails on an empty block:

```python
import numpy as np, librosa
librosa.effects.preemphasis(np.array([]), coef=0.5, zi=np.zeros(1), return_zf=True)
# ValueError: v cannot be empty   (from scipy/signal/_signaltools.py, lfilter FIR path)
```

---

## 5. scipy/scipy : filtfilt(method="gust") et pôles proches de zéro

Posté le 18/11/2026 : https://github.com/scipy/scipy/issues/26755

**Le texte doit être écrit par toi** (politique IA de SciPy, comme pour la section 2). Ci-dessous,
les faits et le code ; à toi la rédaction.

Lien : https://github.com/scipy/scipy/issues/new/choose (Bug report). Un nouveau ticket plutôt
qu'un commentaire sur #26610 : la PR #26624 ferme #26610 en retirant les zéros exacts, et ce cas
n'est pas couvert par elle. Le citer comme lié.

Faits vérifiés le 18/11/2026 sur SciPy 1.18.1, NumPy 2.5.3, Python 3.13
(script complet : `benchmarks/gust_near_zero_poles.py`) :
- Filtre a = poly([0.5, p, p]) (un pôle à 0,5 et un pôle double en p), b = [1], N = 97.
  Quand p tend vers 0, la sortie exacte tend vers celle de a = [1, -0.5] (écart d'environ 3,6 p
  en relatif). Celle de SciPy s'en éloigne : 5,4e-3 en relatif pour p = 1e-7, contre 3,6e-7
  attendu.
- La propriété documentée y_fb == y_bf (et donc le renversement temporel) est cassée : écart
  2,6e-7 (p = 1e-6) et 6,5e-5 (p = 1e-7) en relatif.
- Le problème est bien conditionné (nombre de conditionnement d'environ 1 en x et 2 en `a`,
  mesuré en 80 chiffres). C'est l'algorithme qui perd les chiffres : erreur de 7,3e9 ε pour
  p = 1e-6 contre une référence en précision étendue de l'algorithme de SciPy lui-même.
- Cause : la matrice M du moindre carré de `_filtfilt_gust` devient quasi singulière
  (cond ≈ 5e12 pour p = 1e-6) ; les arrondis du second membre `y_bf - y_fb` sont amplifiés
  par 1/σ_min. Une troncature de la SVD (`cond=` de `lstsq`) ne corrige pas, elle dégrade.
- C'est la version continue de #26610 : p = 0 exactement donne des zéros en fin de `a`.
- Trouvé par la campagne dsp-props (relation : filtfilt gust commute avec le renversement
  temporel) sur un filtre d'ordre 6 à deux pôles à 0,001.
- Pas de correctif évident à proposer : le seuil « coefficient négligeable » n'a pas de valeur
  naturelle ; une formulation mieux conditionnée des états initiaux serait la vraie réparation.
  On peut le dire dans le ticket sans proposer de PR.

Code pour reproduire (sans dépendance autre que SciPy) :

```python
import numpy as np
from scipy.signal import filtfilt

x = np.random.default_rng(0).standard_normal(97)
y0 = filtfilt([1.0], [1.0, -0.5], x, method="gust")   # limit p -> 0
for p in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    a = np.poly([0.5, p, p])
    y = filtfilt([1.0], a, x, method="gust")
    yr = filtfilt([1.0], a, x[::-1], method="gust")[::-1]
    print(f"p={p:.0e}  |y - y0|/|y0| = {np.linalg.norm(y - y0) / np.linalg.norm(y0):.1e}"
          f"   reversal gap = {np.linalg.norm(y - yr) / np.linalg.norm(y0):.1e}")
# |y - y0|/|y0| should be about 3.6 p; it is 7.0e-06 at p=1e-6 and 5.4e-03 at p=1e-7
```

Sortie obtenue :

```
p=1e-02  |y - y0|/|y0| = 3.7e-02   reversal gap = 6.7e-15
p=1e-03  |y - y0|/|y0| = 3.6e-03   reversal gap = 9.2e-13
p=1e-04  |y - y0|/|y0| = 3.6e-04   reversal gap = 1.4e-10
p=1e-05  |y - y0|/|y0| = 3.6e-05   reversal gap = 9.1e-09
p=1e-06  |y - y0|/|y0| = 7.0e-06   reversal gap = 2.6e-07
p=1e-07  |y - y0|/|y0| = 5.4e-03   reversal gap = 6.5e-05
```
