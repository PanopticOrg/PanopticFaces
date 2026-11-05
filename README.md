# PanopticFaces — plugin Panoptic

*[English version](README.en.md)*











  
  
Ce plugin permet de détecter automatiquement des visages dans les corpus [Panoptic](https://github.com/CERES-Sorbonne/Panoptic). Grâce à un système de galerie vous pouvez choisir quels visages détecter et annoter les noms dans des propriétés "multi tags".
> **Cadre légal**  
La reconnaissance faciale traite des données biométriques : le [RGPD](https://www.cnil.fr/fr/biometrie) et le  [règlement européen sur l'IA](https://artificialintelligenceact.eu/fr/) s'appliquent.  
**PanopticFaces** ne publie pas de galerie d'images ni d'embeddings.
---

## Installation

1. Ouvrez Panoptic.
2. Ajoutez le plugin avec l'adresse de ce dépôt GitHub (ou le dossier sur votre ordinateur)
3. Le plugin télécharge le modèle `buffalo_l` (environ 280 Mo) automatiquement à la première exécution de **detect_faces**.

## Utilisation
Dans la vue grille : Exécuter -> **detect_faces**. À exécuter avant toute autre fonction du plugin. Avant de continuer attendez que la tâche soit complètement finie.
**PanopticFaces** expose plusieurs types de fonctions dans **Panoptic**.

- Clustering : **cluster_faces** permet de regrouper les photos par similarité de visage.
- Similarité : **find_similar_faces** renvoie les images les plus similaires à celle sélectionnée (typiquement dans la Modal Image de Panoptic) en prenant en compte les visages et non l'image entière.
- Exécuter : 
  -  **import_gallery** permet d'importer une galerie de visages sous forme de dossier qui contient un sous-dossier par
     personne (exemple : `galerie/Marie Curie/*.jpg`)
  -  **add_reference** pour ajouter manuellement une personne à la galerie en indiquant son nom et un dossier contenant des images.
  - **forget_faces** : efface les visages des images sélectionnées et vide leurs
  propriétés. Cochez `everything` pour tout effacer.
  - **remove_person** : retire une personne de la galerie.
  - Pour tout effacer d'un coup : supprimez le dossier `plugin_data/faces/` du projet
    Panoptic.
  - Si vous avez construit une galerie : supprimez aussi les dossiers `gallery/` et
    `gallery_cache/` quand vous n'en avez plus besoin.

## Ce que le plugin ajoute dans Panoptic

Les résultats apparaissent dans le groupe de propriétés **Faces** :

| Propriété     | Type       | Signification                                                  |
|---------------|------------|----------------------------------------------------------------|
| Persons       | tags       | les personnes reconnues dans l'image                           |
| Party         | tags       | le parti actuel de ces personnes (si la galerie le donne)      |
| Faces         | nombre     | le nombre de visages trouvés                                   |
| Unknown faces | nombre     | les visages que la galerie n'a pas reconnus                    |
| Person score  | nombre     | la reconnaissance la moins sûre de l'image (triez pour vérifier) |


## Construire votre galerie

La galerie est la collection d'images de référence servant à détecter les noms des personnes identifiées. Elle n'est pas utile aux fonctions de clustering et de similarité qui se basent entièrement sur les photos et n'ont pas besoin de lier l'image à une personne.

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
- Utilisez des images contenant une seule personne bien visible. Type portrait idéalement.
- Formats acceptés : JPG, PNG, WEBP, BMP, TIFF.

### Méthode 1 : à la main

1. Créez le dossier et les sous-dossiers comme ci-dessus.
2. Mettez-y des photos que vous avez le droit d'utiliser.
3. Dans Panoptic, lancez **import_gallery** et choisissez le dossier.

Sans dossier, vous pouvez aussi sélectionner des images dans Panoptic et lancer
**add_reference** avec le nom de la personne.

### Méthode 2 : avec l'outil automatique (personnalités publiques)

L'outil `tools/build_gallery.py` télécharge des photos de personnalités publiques
depuis des sources ouvertes.

1. Ouvrez un terminal dans le dossier du plugin.
2. Lancez l'outil avec le Python de Panoptic :

       python tools/build_gallery.py all --contact <votre url ou e-mail>

   `--contact` ajoute votre contact aux requêtes. Sans lui, Wikimedia ralentit
   beaucoup les téléchargements. Pour un premier essai, ajoutez `--limit 50` (les
   50 personnes les plus connues seulement).
3. Attendez. La première fois, cela peut prendre plusieurs heures. Si vous arrêtez,
   relancez la même commande : l'outil reprend où il en était.
4. La galerie est créée dans le dossier `gallery/`. Dans Panoptic, lancez
   **import_gallery** sur ce dossier.

Ce que fait l'outil :

- **Qui ?** Des personnalités publiques : élus français (députés depuis 2012,
  sénateurs, maires des villes de plus de 30 000 habitants), personnalités françaises
  et mondiales connues.
- **D'où viennent les photos ?** Wikidata, Wikimedia Commons, et les portraits
  officiels de l'Assemblée nationale et du Sénat.
- **Vérification :** chaque photo est comparée au portrait officiel ou au portrait
  Wikidata. Les photos d'autres personnes sont retirées.
- **Fichiers créés :** `persons.csv` (les personnes et leur parti), `manifest.csv` (la
  source et la licence de chaque photo) et `conflicts.csv` (les personnes qui se
  ressemblent, à vérifier).

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
