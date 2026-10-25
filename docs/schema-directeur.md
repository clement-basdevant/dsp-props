# Schéma directeur

Le schéma directeur complet (état de l'art, catalogue de propriétés, feuille de route, pistes de publication) est maintenu dans le document de projet : https://claude.ai/code/artifact/a8c9b15e-7a21-480b-9455-3732a46f1e03

## Résumé

- **Objectif** : une couche DSP au-dessus d'Hypothesis : relations métamorphiques, générateurs de signaux, tolérances par opérateur.
- **Périmètre v1** : NumPy/SciPy, LTI, transformées, filtrage par blocs, multicadence.
- **Critère de réussite** : score de mutation, temps de détection, taille du contre-exemple minimal.
- **Publications visées** : JOSS (logiciel), SciPy/EuroSciPy (talk), ICST ou atelier MET (catalogue + évaluation par mutation), GRETSI/EUSIPCO (moteur de tolérance).

## Phase 1 : état

- [x] Squelette du paquet, licence, CI
- [x] Générateurs : impulsion, DC, sinusoïde hors grille, quasi-Nyquist, tons proches, parcimonieux, bruit blanc ; longueurs pathologiques
- [x] Relations : linéarité, décalage circulaire, Parseval, symétrie hermitienne, inversibilité, théorème de convolution
- [x] `numpy.fft` passe toutes les relations ; 8 mutants sur 8 détectés
- [x] Premier bug réel signalé en amont : [scipy/scipy#26406](https://github.com/scipy/scipy/issues/26406) (`lfilter`/`sosfilt` sur bloc vide), corrigé par notre PR [scipy/scipy#26415](https://github.com/scipy/scipy/pull/26415), fusionnée le 23/10/2026 (SciPy 2.0.0)
- [x] Transformées réelles (`kind="rdft"`), transformées génériques (DCT, STFT) et streaming par blocs (`dp.streaming`, `dp.stateful`)
- [x] 20 mutants en quatre familles, tous détectés

## Phase 2 : état

- [x] Module de calibration (`dp.calibrate`) : erreur mesurée contre une FFT en long double, longueur par longueur, avec plancher analytique c · ε · log2 N
- [x] La calibration refuse un opérateur faux (erreur > 10^6 ε) au lieu d'absorber le bug
- [x] Profil d'erreur sur N = 2…1200 (`benchmarks/error_profile.csv`) : pire cas NumPy 4,1 ε, SciPy 3,7 ε, PyTorch/MKL 195 ε (N = 596 = 4 × 149). Pour torch, 220 longueurs dépassent 20 ε ; toutes ont un facteur premier entre 17 et 149, et 130 d'entre elles sont multiples de 4.
- [x] Modèle d'erreur IIR (`IIRTolerance`) : borne en ε · ordre · ‖h_A‖₁ · (‖b‖₁ + (‖a‖₁ − 1) ‖h_BA‖₁) · ‖x‖, jamais dépassée sur 80 filtres de rayon 0,5 à 0,999 et d'ordre 2 à 12 (pire rapport 0,008), et qui détecte toujours la récursion en float32
- [x] Figure erreur/N pour l'article (`docs/figures/error_profile.png`)

## Phase 3 : état

- [x] Relations LTI : linéarité, invariance temporelle, causalité (`dp.lti_filter`)
- [x] Cohérence fonction de transfert / réponse mesurée (`transfer=(b, a)`)
- [x] Famille de 5 mutants IIR, tous détectés ; sans fonction de transfert déclarée, le filtre « faux mais LTI » reste un angle mort documenté
- [x] Stabilité EBSB avec gain déclaré (`max_gain`) : entrée pire cas sign(h[N-1-k]) construite à partir de la réponse impulsionnelle mesurée
- [x] Multicadence (`dp.resampler(L, M)`) : linéarité, invariance périodique (retard M en entrée = retard L en sortie), longueur de sortie, gain DC, rejet du repliement ; `resample_poly` passe pour 1/2, 3/1, 2/3, 5/4, 1/4, 3/8
- [x] Famille de 6 mutants multicadence, tous détectés ; une décimation sans filtre (`x[::2]`) passe tout sauf le rejet du repliement
- [x] L'invariance temporelle accepte les filtres non causaux (comparaison après le retard)

## Phase 5 : état

- [x] Chaque échec garde l'entrée minimale trouvée par Hypothesis et le nombre d'entrées générées avant le premier échec
- [x] Rapport exportable en JSON et Markdown, avec le temps par relation
- [x] Mesures sur les 31 mutants (`benchmarks/mutant_results.json`) : contre-exemple de 3 échantillons au plus pour 27 sur 30 (médiane 2) ; 27 mutants détectés dès les 3 premières entrées, le plus lent après 32
- [x] Rapport HTML autonome (`report.to_html()`, `dsp-props check --html`), lisible en thème clair et sombre

## Phase 6 : état

- [x] Outil en ligne de commande `dsp-props check` : découverte automatique des opérateurs déclarés (module, fichier, `module:attribut`, registres en dictionnaire), export JSON et Markdown, code de sortie pour la CI
- [x] Adaptateur pour du code C (`dsp_props.adapters`, ctypes) : fonctions sans état et traitement par blocs avec état ; un biquad C qui perd une variable d'état entre blocs passe les relations LTI sur un appel unique et n'est détecté que par l'invariance au découpage

## Observations à suivre en phase 2

- La tolérance FFT (c · ε · log2 N · ‖y‖, c = 20) a rejeté le calcul de référence du théorème du décalage tant que l'argument des twiddles n'était pas réduit modulo N : la référence elle-même doit être plus précise que l'opérateur testé.
- Une DFT naïve en O(N²) dépasse la borne FFT : le modèle de tolérance doit dépendre de l'algorithme, pas seulement de l'opération.
- Les amplitudes extrêmes (1e±100) imposent des normes calculées sans débordement.
- La première CI a trouvé un cas réel : une entrée quasi dénormalisée (‖x‖ ≈ 3e-303) faisait produire NaN au calcul de norme complexe (division complexe par un sous-normal). Les dénormaux doivent devenir un générateur à part entière.
