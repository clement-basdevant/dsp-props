# Publier une version sur PyPI

Le paquet est prêt (version dans `src/dsp_props/__init__.py`, actuellement 0.1.3).
`python -m build && python -m twine check dist/*` passe. La publication est déclenchée par un tag ;
rien n'est publié tant qu'aucun tag `v*` n'est poussé.

## Une seule fois : déclarer GitHub comme éditeur de confiance sur PyPI

1. Créer un compte sur https://pypi.org et activer la double authentification.
2. Aller dans *Your account > Publishing > Add a new pending publisher* et remplir :
   - PyPI project name : `dsp-props`
   - Owner : `clement-basdevant`, Repository : `dsp-props`
   - Workflow name : `release.yml`, Environment : `pypi`
3. Sur GitHub, *Settings > Environments > New environment* : `pypi`.

## À chaque version

1. Mettre à jour `__version__` dans `src/dsp_props/__init__.py` et `version` dans `CITATION.cff`.
2. Vérifier que la CI est verte.
3. Créer et pousser le tag :

   ```
   git tag v0.1.0
   git push origin v0.1.0
   ```

On peut aussi lancer la publication à la main : onglet *Actions > Release > Run workflow*.

Le workflow `Release` construit le paquet, le vérifie et le publie. Une version publiée sur PyPI
ne peut plus être remplacée (seulement retirée) : vérifier le numéro avant de pousser le tag.
