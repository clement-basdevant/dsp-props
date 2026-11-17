# dsp-props : critique et pistes pour un outil utile

17 novembre 2026 · Clément Basdevant

## Diagnostic d'ensemble

dsp-props est solide sur le plan scientifique, mais il n'est encore utile qu'à son auteur. Toutes les preuves de valeur viennent du dépôt lui-même : on n'a pas encore d'utilisateur extérieur, pas de ticket ouvert par un tiers, ni de projet qui l'ait adopté.

Ce qui est acquis est rare pour un projet de six semaines :

- un catalogue de relations couvrant transformées, LTI, multicadence et streaming, avec des preuves Coq en annexe ;
- un moteur de tolérance honnête sur ses limites (`docs/blind-spots.md`) ;
- une calibration qui attrape une FFT dégradée dès 21 ε, sans fausse alarme sur 12 implémentations correctes ;
- des résultats réels en amont : un correctif fusionné dans SciPy (scipy#26415), un autre en revue (scipy#26624) ;
- des contre-exemples minuscules (médiane de 2 échantillons).

La faiblesse centrale est la circularité de la validation. Les 31 mutants sont écrits à la main dans le vocabulaire des relations, et les benchmarks mesurent l'outil contre lui-même. Pour devenir utile, le projet doit maintenant être éprouvé hors de chez lui : par du code qu'il n'a pas choisi, des bugs qu'il n'a pas inventés et des gens qui ne l'ont pas écrit. Le reste de ce document se range derrière cette priorité.

## Critique 1 : la tolérance par défaut est le maillon faible

La tolérance par défaut laisse passer des fautes que la calibration attrape 7 à 3400 fois plus tôt (sur 5 filtres et 4 rapports). C'est le cœur de la promesse (« une tolérance dérivée d'un modèle d'erreur »), et elle n'est tenue que pour l'IIR.

- **Constantes d'ingénierie.** c = 20 et c = 1000 ne sont pas dérivés. Une FFT à 90 ε passe par défaut ; une FFT à facteurs de rotation récursifs (38 ε) aussi.
- **Modèle IIR pessimiste.** Le seul modèle dérivé est 7 à 3400 fois plus large que l'erreur réelle mesurée. Il est sûr, mais il n'attrape rien en dessous de 7e5 ε.
- **Angles morts structurels.** Les coefficients arrondis d'un rééchantillonneur ne sont jamais détectés. Un filtre faux mais LTI et causal passe si l'on ne déclare pas sa fonction de transfert.
- **Trou entre 20 ε log2 N et 1e6 ε.** `dp.calibrate` refuse au-delà de 1e6 ε, mais accepte tout ce qui est en dessous comme « normal ».

Ce qu'il faudrait faire :

1. **Faire de la calibration le chemin par défaut**, pas une option. On calibre une fois l'implémentation de confiance, on enregistre le profil dans un fichier versionné (`tolerances.json`), et les tests suivants s'y comparent. C'est aussi le cas d'usage le plus vendable : **détecter une régression de précision entre deux versions** d'une bibliothèque, ce que personne ne fait aujourd'hui.
2. **Resserrer le modèle IIR** en tenant compte de la forme réelle (SOS, forme directe II transposée) plutôt que de la seule forme directe I.
3. **Dériver un modèle pour les rééchantillonneurs et le streaming**, ou documenter clairement que c = 1000 est un garde-fou contre les fautes grossières uniquement.
4. **Ajouter une relation de cohérence float32/float64** : elle comble l'angle mort des coefficients arrondis, car un opérateur en float64 doit être nettement plus proche d'une référence que le même en float32.
5. **Faire des dénormaux un générateur à part entière**, puisque la première CI a déjà trouvé un NaN sur une norme de 3e-303.

## Critique 2 : l'outil ne s'intègre pas là où les gens testent

Un développeur DSP teste avec pytest, dans la suite de tests existante de son projet. dsp-props lui demande d'adopter une API à part (`dp.verify`, une CLI). Le paquet affiche le classificateur `Framework :: Pytest`, mais il n'a pas de plugin pytest (pas de point d'entrée `pytest11`).

- **Plugin pytest.** Un marqueur ou une fixture qui transforme chaque opérateur déclaré en tests pytest, une relation = un test. Ainsi l'échec apparaît dans le rapport habituel, avec le contre-exemple.
- **Tests de non-régression générés.** C'est déjà une piste ouverte, et la plus rentable : quand une relation échoue, écrire un fichier `test_regression_<op>.py` qui rejoue le contre-exemple minimal, sans Hypothesis. Le développeur le committe et garde le bug sous contrôle.
- **Opérateurs paramétrés.** Aujourd'hui on déclare un filtre précis. Les bibliothèques testent des familles (`butter(order, Wn)`, `resample_poly(x, up, down)`). Il faut pouvoir tirer aussi les paramètres, avec des stratégies de conception de filtres (pôles près du cercle, ordres élevés, rapports premiers entre eux).
- **Array API.** En visant le standard Array API, une seule déclaration testerait NumPy, CuPy, PyTorch et JAX, CPU et GPU. Les FFT GPU sont exactement là où les écarts de précision sont mal connus (cf. les 195 ε de PyTorch sur CPU).
- **Coût.** La suite interne prend environ 5 minutes. Il faut un profil « rapide » (30 exemples, quelques secondes) pour la CI quotidienne et un profil « nuit » pour la chasse aux bugs.
- **Documentation orientée tâche.** Le tutoriel fait 455 lignes. Il manque une page « tester votre filtre en 5 minutes » et une galerie de cas réels (lfilter sur bloc vide, gust, PyTorch FFT), chacun avec le code qui le trouve.

Tant que ces points manquent, l'outil reste une démonstration de recherche plutôt qu'un outil de travail.

## Critique 3 : la validation est circulaire et la dette s'accumule

Le score de 31/31 mutants mesure que les relations attrapent les fautes pour lesquelles elles ont été conçues. Un relecteur le verra tout de suite. Il faut une preuve indépendante.

1. **Corpus de bugs historiques.** Rejouer de vrais bugs passés de `numpy.fft`, `scipy.signal`, librosa et torchaudio, retrouvés dans leurs trackers (ticket corrigé + version fautive). Mesure : combien sont trouvés, en combien d'entrées, avec quelle taille de contre-exemple. C'est l'évaluation que demandent ICST ou l'atelier MET, et elle ne dépend pas de nos choix.
2. **Mutation automatique.** Passer mutmut ou cosmic-ray sur du code DSP en Python pur (FFT radix-2, biquad, polyphase) et comparer le score des relations à celui des tests existants de ces projets. Cela démontre la valeur ajoutée par rapport à `assert_allclose`.
3. **Comparaison aux suites de tests réelles.** `survey_test_suites.py` montre déjà que `numpy.fft` n'a aucun test Hypothesis et que 56 de ses tolérances valent 1e-6 en absolu. C'est un argument fort, à pousser : ces suites attraperaient-elles nos mutants ?

Dette à solder avant toute publication :

- `benchmarks/validation.py --only false-positives` est cassé depuis #10 (TypeError sur `precision=`), et les 2000 exécutions sans fausse alarme datent d'avant #7 à #10. Le chiffre du papier n'est donc plus reproductible.
- La PR #15 affirme le contraire dans CLAUDE.md. Il faut vérifier, corriger, puis fusionner ou fermer.
- `benchmarks/campaign.py` garde sa propre copie de `block_splits` plafonnée à 9 blocs, contre max(N, 16) dans la bibliothèque.
- Le papier JOSS et l'article datent du 25/10 : les résultats M-1 et A9 du 9/11 n'y sont pas.
- Les chiffres changent à chaque PR et sont recopiés à la main dans le README, CLAUDE.md et les deux papiers. Un script qui génère un fichier `results.tex`/`results.md` à partir des JSON, vérifié en CI, éviterait les incohérences.

## Critique 4 : publier, mais sur la bonne preuve

Le projet vise quatre publications. Chacune bute aujourd'hui sur le même manque : l'usage et l'évaluation externes.

- **JOSS.** Les relecteurs regardent l'utilité pour la recherche, l'historique de développement et la communauté. Un dépôt de six semaines, un seul contributeur et aucun utilisateur tiers, c'est le point faible. Un ou deux usages réels (un labo, un projet open source qui l'intègre dans sa CI) pèseraient plus que n'importe quel nouveau chiffre. (À vérifier dans les critères JOSS actuels avant de soumettre.)
- **Article ICST/MET.** Avec le corpus de bugs historiques (critique 3), la contribution devient « des relations métamorphiques DSP retrouvent X bugs réels sur Y ». Sans lui, c'est un catalogue évalué contre ses propres mutants.
- **GRETSI/EUSIPCO.** Le résultat le plus original est là : les tolérances fixes (1e-7, 1e-6) n'attrapent aucune faute de précision, la calibration les attrape dès 21 ε, et le modèle IIR dérivé est trop pessimiste. C'est un papier court et autonome, publiable rapidement.
- **SciPy/EuroSciPy.** Un talk sur les bugs trouvés en amont, avec la démo pytest, servirait surtout à recruter des utilisateurs.

Le meilleur canal d'adoption est l'amont lui-même. Après scipy#26415, proposer à SciPy quelques tests de propriété pour `scipy.signal` (bloc vide, invariance au découpage avec `zi`) installerait l'outil là où il a déjà prouvé sa valeur. Il faut respecter leur politique IA : texte et code rédigés par l'auteur. scipy#26624 (gust), scipy#26417 (tf2sos) et pytorch/audio#4238 attendent encore un mainteneur ; une relance polie après deux semaines est raisonnable.

## Feuille de route priorisée

L'ordre suit une règle simple : d'abord solder la dette, ensuite rendre l'outil adoptable, enfin produire la preuve externe qui nourrit les publications.

### D'ici un mois : remettre le socle d'aplomb

- [ ] Réparer `validation.py` (argument `precision=`) et relancer les 2000 exécutions de fausses alarmes sur le main actuel
- [ ] Trancher la PR #15, puis la fusionner ou la fermer
- [ ] Supprimer la copie de `block_splits` dans `campaign.py` et relancer la campagne
- [ ] Script qui génère les chiffres du papier depuis les JSON, vérifié en CI
- [ ] Reporter M-1 et A9 dans le papier JOSS et l'article
- [ ] Plugin pytest minimal (point d'entrée `pytest11`, une relation = un test)

### De un à trois mois : rendre l'outil adoptable

- [ ] Génération de tests de non-régression à partir des contre-exemples
- [ ] Calibration par défaut avec profil versionné, et mode « régression de précision entre versions »
- [ ] Relation float32/float64 et générateur de dénormaux
- [ ] Opérateurs paramétrés (stratégies de conception de filtres)
- [ ] Page « tester votre filtre en 5 minutes » et galerie de cas réels
- [ ] Soumettre le papier court sur les tolérances (GRETSI ou EUSIPCO)

### De trois à six mois : la preuve externe

- [ ] Corpus de bugs historiques (NumPy, SciPy, librosa, torchaudio) et campagne de redécouverte
- [ ] Mutation automatique (mutmut) comparée aux suites de tests existantes
- [ ] Support Array API (CuPy, JAX, PyTorch GPU) et profil d'erreur des FFT GPU
- [ ] Proposer des tests de propriété à `scipy.signal` ; trouver un ou deux utilisateurs externes
- [ ] Soumettre JOSS avec ces usages, puis l'article ICST/MET avec le corpus de bugs

## Ce qu'il ne faut pas faire

- **Ajouter des relations avant d'avoir des utilisateurs.** Le catalogue est déjà large ; les identités nobles ou les estimateurs peuvent attendre qu'un utilisateur les demande.
- **Écrire d'autres mutants à la main pour gonfler le score.** Chaque mutant de plus renforce la circularité au lieu de la réduire.
- **Élargir une tolérance pour faire taire une alarme.** C'est la règle du projet, et c'est aussi ce qui rend l'outil crédible.
- **Soumettre JOSS trop tôt.** Un refus pour manque d'impact coûte plus cher que deux mois d'attente avec un vrai utilisateur.
- **Multiplier les chiffres recopiés à la main.** Tant qu'ils ne sont pas générés, chaque PR risque d'en désynchroniser un.

Document de travail : https://claude.ai/artifact/NdrJAUkw9zR8C7dWsbN1mM
