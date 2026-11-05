# PanopticFaces — Panoptic plugin

*[Version française](README.md)*

This plugin automatically detects faces in [Panoptic](https://github.com/CERES-Sorbonne/Panoptic) corpora. With a gallery system, you can choose which faces to detect and write the names into "multi tags" properties.
> **Legal framework**  
Face recognition processes biometric data: the [GDPR](https://www.cnil.fr/fr/biometrie) and the [EU AI Act](https://artificialintelligenceact.eu/) apply.  
**PanopticFaces** does not publish any image gallery or embeddings.
---

## Installation

1. Open Panoptic.
2. Add the plugin with the address of this GitHub repository (or the folder on your computer).
3. The plugin automatically downloads the `buffalo_l` model (about 280 MB) the first time **detect_faces** runs.

## Usage
In the grid view: Execute -> **detect_faces**. Run it before any other function of the plugin. Wait until the task is completely finished before you continue.
**PanopticFaces** provides several types of functions in **Panoptic**.

- Clustering: **cluster_faces** groups photos by face similarity.
- Similarity: **find_similar_faces** returns the images most similar to the selected one (typically in the Panoptic Image Modal), based on the faces and not the whole image.
- Execute:
  -  **import_gallery** imports a face gallery from a folder that contains one sub-folder per
     person (example: `gallery/Marie Curie/*.jpg`).
  -  **add_reference** manually adds a person to the gallery, from a name and a folder of images.
  - **forget_faces**: erases the faces of the selected images and clears their
  properties. Check `everything` to erase everything.
  - **remove_person**: removes a person from the gallery.
  - To erase everything at once: delete the `plugin_data/faces/` folder of the Panoptic
    project.
  - If you built a gallery: also delete the `gallery/` and `gallery_cache/` folders
    when you no longer need them.

## What the plugin adds to Panoptic

Results appear in the **Faces** property group:

| Property      | Type       | Meaning                                                        |
|---------------|------------|----------------------------------------------------------------|
| Persons       | tags       | the people recognized in the image                             |
| Party         | tags       | the current party of these people (if the gallery provides it) |
| Faces         | number     | the number of faces found                                      |
| Unknown faces | number     | the faces the gallery did not recognize                        |
| Person score  | number     | the least certain recognition in the image (sort to review)    |


## Building your gallery

The gallery is the collection of reference images used to detect the names of identified people. It is not needed for the clustering and similarity functions, which rely only on the photos and do not need to link an image to a person.

A gallery is a simple folder, with one sub-folder per person:

    my_gallery/
    ├── Marie Curie/
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── 3.jpg
    └── Victor Hugo/
        ├── 1.jpg
        └── 2.jpg

- The sub-folder name is the name shown in Panoptic.
- Use 5 to 20 varied photos per person (ages, angles, lighting).
- Use images with a single, clearly visible person. Ideally portraits.
- Accepted formats: JPG, PNG, WEBP, BMP, TIFF.

### Method 1: by hand

1. Create the folder and sub-folders as shown above.
2. Put in photos you have the right to use.
3. In Panoptic, run **import_gallery** and choose the folder.

Without a folder, you can also select images in Panoptic and run
**add_reference** with the person's name.

### Method 2: with the automatic tool (public figures)

The `tools/build_gallery.py` tool downloads photos of public figures
from open sources.

1. Open a terminal in the plugin folder.
2. Run the tool with Panoptic's Python:

       python tools/build_gallery.py all --contact <your url or e-mail>

   `--contact` adds your contact to the requests. Without it, Wikimedia slows
   downloads down a lot. For a first try, add `--limit 50` (only the
   50 best-known people).
3. Wait. The first run can take several hours. If you stop it,
   run the same command again: the tool resumes where it stopped.
4. The gallery is created in the `gallery/` folder. In Panoptic, run
   **import_gallery** on this folder.

What the tool does:

- **Who?** Public figures: French elected officials (deputies since 2012,
  senators, mayors of cities with more than 30,000 inhabitants), well-known French
  and international figures.
- **Where do the photos come from?** Wikidata, Wikimedia Commons, and the official
  portraits of the Assemblée nationale and the Sénat.
- **Verification:** each photo is compared to the official portrait or the Wikidata
  portrait. Photos of other people are removed.
- **Files created:** `persons.csv` (the people and their party), `manifest.csv` (the
  source and licence of each photo) and `conflicts.csv` (people who look
  alike, to review).

## Settings

- `match_threshold` (0.45): higher = fewer wrong names, but fewer faces
  recognized.
- `match_margin` (0.05): gives no name if two people in the gallery look
  too similar.
- `det_size` (640): set to 1024 to find small faces in a crowd.
- `compute_on_import` (disabled): automatically analyzes new images.

## Licence

- The code is licensed under the **GNU GPL v3**, like Panoptic. See the
  [LICENSE](LICENSE) file.
- The InsightFace models (`buffalo_l`, `buffalo_s`, `antelopev2`) are restricted to
  **non-commercial research use**.
