# Tickets à poster

Brouillons prêts, à poster depuis le compte GitHub du projet. Une fois posté, remplacer « à poster » par le lien.

| # | Où | Statut |
| --- | --- | --- |
| 0 | scipy/scipy : lfilter/sosfilt sur bloc vide | posté : [scipy#26406](https://github.com/scipy/scipy/issues/26406), corrigé par [scipy#26415](https://github.com/scipy/scipy/pull/26415) (fusionnée le 23/10/2026, SciPy 2.0.0) |
| 1 | pytorch/audio : deemphasis écrête à [-1, 1] | posté : [pytorch/audio#4238](https://github.com/pytorch/audio/issues/4238) |
| 2 | scipy/scipy : tf2sos supprime les retards purs | posté : [scipy#26417](https://github.com/scipy/scipy/issues/26417) |
| 3 | commentaire sur scipy#26406 : impact librosa | posté le 15/10/2026 sur [scipy#26406](https://github.com/scipy/scipy/issues/26406) |

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
