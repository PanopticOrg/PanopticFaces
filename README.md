# Faces — plugin Panoptic

*[English version](README.en.md)*

Ce plugin trouve les visages dans vos images, les reconnaît et les regroupe.
Il fonctionne dans [Panoptic](https://github.com/CERES-Sorbonne/Panoptic) et utilise
InsightFace pour l'analyse des visages.

> **Cadre légal**  
La reconnaissance faciale traite des données biométriques :  
le [RGPD](https://www.cnil.fr/fr/biometrie) et le  [règlement européen sur l'IA](https://artificialintelligenceact.eu/fr/) s'appliquent. Renseignez-vous avant de l'utiliser.
---

## Installation

1. Ouvrez Panoptic.
2. Ajoutez le plugin avec l'adresse de ce dépôt GitHub (ou le dossier sur votre
   ordinateur).
3. Les dépendances (`insightface`, `onnxruntime`, `onnx`, `numpy`, `pillow`) sont
   listées dans `requirements.txt`.

Au premier lancement, le plugin télécharge le modèle `buffalo_l` (environ 280 Mo).

## Utilisation

1. **detect_faces** : cherche les visages dans les images choisies.
2. Créez une **galerie** de personnes connues, au choix :
   - choisissez des portraits d'une personne et lancez **add_reference** avec son nom ;
   - ou lancez **import_gallery** sur un dossier qui contient un sous-dossier par
     personne (exemple : `galerie/Marie Curie/*.jpg`).

   Chaque photo donne son plus grand visage. 5 à 20 photos variées par personne
   suffisent.
3. Après chaque changement de la galerie, tous les visages sont reconnus à nouveau.
4. **cluster_faces** regroupe les visages inconnus qui se ressemblent. Donnez un nom à
   un groupe, sélectionnez-le et lancez **add_reference** pour l'ajouter à la galerie.
5. **find_similar_faces** trouve les images où apparaît un visage proche de celui
   sélectionné.

## Ce que le plugin ajoute dans Panoptic

Les résultats apparaissent dans le groupe de propriétés **Faces** :

| Propriété     | Type       | Signification                                                  |
|---------------|------------|----------------------------------------------------------------|
| Persons       | tags       | les personnes reconnues dans l'image                           |
| Party         | tags       | le parti actuel de ces personnes (si la galerie le donne)      |
| Faces         | nombre     | le nombre de visages trouvés                                   |
| Unknown faces | nombre     | les visages que la galerie n'a pas reconnus                    |
| Person score  | nombre     | la reconnaissance la moins sûre de l'image (triez pour vérifier) |

## Supprimer des données

- **forget_faces** : efface les visages des images sélectionnées et vide leurs
  propriétés. Cochez `everything` pour tout effacer.
- **remove_person** : retire une personne de la galerie.
- Pour tout effacer d'un coup : supprimez le dossier `plugin_data/faces/` du projet
  Panoptic.
- Si vous avez construit une galerie : supprimez aussi les dossiers `gallery/` et
  `gallery_cache/` quand vous n'en avez plus besoin.

Ces actions permettent de répondre à une demande d'effacement (article 17 du RGPD).

## Construire votre galerie

La galerie, ce sont les personnes que le plugin sait reconnaître. **Aucune galerie
n'est fournie** : chacun construit la sienne, sur son ordinateur, pour son propre
projet. Ne la partagez pas (voir plus bas).

Une galerie est un simple dossier, avec un sous-dossier par personne :

    ma_galerie/
    ├── Marie Curie/
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── 3.jpg
    └── Victor Hugo/
        ├── 1.jpg
        └── 2.jpg

- Le nom du sous-dossier est le nom affiché dans Panoptic.
- Mettez 5 à 20 photos par personne, variées (âges, angles, lumières).
- Sur chaque photo, la personne doit avoir **le plus grand visage**. Les portraits
  sont parfaits.
- Formats acceptés : JPG, PNG, WEBP, BMP, TIFF.

### Méthode 1 : à la main

1. Créez le dossier et les sous-dossiers comme ci-dessus.
2. Mettez-y des photos que vous avez le droit d'utiliser.
3. Dans Panoptic, lancez **import_gallery** et choisissez le dossier.

Sans dossier, vous pouvez aussi sélectionner des images dans Panoptic et lancer
**add_reference** avec le nom de la personne.

### Méthode 2 : avec l'outil automatique (personnalités publiques)

L'outil `tools/build_gallery.py` télécharge des photos de personnalités publiques
depuis des sources ouvertes, puis garde les bonnes.

1. Ouvrez un terminal dans le dossier du plugin.
2. Lancez l'outil avec le Python de Panoptic (il contient déjà InsightFace) :

       python tools/build_gallery.py all --contact <votre url ou e-mail>

   `--contact` ajoute votre contact aux requêtes. Sans lui, Wikimedia ralentit
   beaucoup les téléchargements. Pour un premier essai, ajoutez `--limit 50` (les
   50 personnes les plus connues seulement).
3. Attendez. La première fois, cela peut prendre plusieurs heures. Si vous arrêtez,
   relancez la même commande : l'outil reprend où il en était.
4. La galerie est créée dans le dossier `gallery/`. Dans Panoptic, lancez
   **import_gallery** sur ce dossier.
5. Quand c'est fini, **supprimez le dossier `gallery_cache/`**. Il contient des
   données personnelles (dates de naissance des élus) qui ne servent plus.

Ce que fait l'outil :

- **Qui ?** Des personnalités publiques : élus français (députés depuis 2012,
  sénateurs, maires des villes de plus de 30 000 habitants), personnalités françaises
  et mondiales connues. **Seulement des adultes.**
- **D'où viennent les photos ?** Wikidata, Wikimedia Commons, et les portraits
  officiels de l'Assemblée nationale et du Sénat.
- **Vérification :** chaque photo est comparée au portrait officiel ou au portrait
  Wikidata. Les photos d'autres personnes sont retirées.
- **Fichiers créés :** `persons.csv` (les personnes et leur parti), `manifest.csv` (la
  source et la licence de chaque photo) et `conflicts.csv` (les personnes qui se
  ressemblent, à vérifier).

## Ce qu'il ne faut jamais partager

Ne publiez jamais, ni sur GitHub, ni par e-mail, ni sur un disque partagé :

| Quoi | Où | Pourquoi |
|------|----|----------|
| La galerie | `gallery/` ou votre dossier | photos de personnes, liées à leur nom et à leur parti |
| Le cache de l'outil | `gallery_cache/` | photos et dates de naissance |
| Les empreintes de visage | `plugin_data/faces/faces.db` dans le projet Panoptic | données biométriques |

Une **empreinte de visage** (en anglais *embedding*) est une suite de 512 nombres
calculée à partir d'un visage. Elle permet de reconnaître la personne sur d'autres
photos : c'est une donnée biométrique, aussi sensible que la photo elle-même. Elle ne
doit **jamais** quitter votre ordinateur, même sans les photos.

**Vous partagez un projet Panoptic ?** Supprimez d'abord le dossier
`plugin_data/faces/` du projet (ou lancez **forget_faces** avec `everything` coché).

Ce que vous pouvez partager : le code de ce dépôt et la méthode. Chacun construit
alors sa propre galerie.

## Réglages

- `match_threshold` (0,45) : plus haut = moins d'erreurs de nom, mais moins de visages
  reconnus.
- `match_margin` (0,05) : ne donne pas de nom si deux personnes de la galerie se
  ressemblent trop.
- `det_size` (640) : passez à 1024 pour trouver les petits visages dans une foule.
- `compute_on_import` (désactivé) : analyse automatiquement les nouvelles images.

## Licence

- Le code est sous licence **GNU GPL v3**, comme Panoptic. Voir le fichier
  [LICENSE](LICENSE).
- Les modèles InsightFace (`buffalo_l`, `buffalo_s`, `antelopev2`) sont réservés à un
  **usage non commercial de recherche**.
- Les photos d'une galerie gardent leur propre licence, indiquée dans `manifest.csv`.
