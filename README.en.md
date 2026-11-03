# Faces — Panoptic plugin

*[Version française](README.md)*

Detects, identifies and clusters faces with InsightFace (RetinaFace + ArcFace).

Panoptic stores one vector per image and vector type, so this plugin keeps its own face
store (`plugin_data/faces/faces.db`: one row per face, with its box, detector score and
512-d embedding). Results reach Panoptic as properties in the **Faces** group:

| Property        | Type       | Meaning                                              |
|-----------------|------------|------------------------------------------------------|
| Persons         | multi_tags | named people in the image                            |
| Party           | multi_tags | current party of the named people (from the gallery's persons.csv) |
| Faces           | number     | detected faces                                       |
| Unknown faces   | number     | faces the gallery could not name                     |
| Person score    | number     | weakest naming similarity in the image — sort by it to review |

## Workflow

1. **detect_faces** on the project (downloads `buffalo_l`, ~280 MB, on first use).
2. Build the gallery, either way:
   - select portraits of one person and run **add_reference** with their name;
   - **import_gallery** on a folder with one sub-folder per person
     (`gallery/Emmanuel Macron/*.jpg`) — e.g. official portraits + Wikimedia Commons.
   Each photo contributes its largest face. 5–20 varied photos per person is plenty.
3. Every change to the gallery re-identifies all stored faces (no model needed).
4. **cluster_faces** groups the faces nobody could name; name a cluster, select it and
   run add_reference to grow the gallery. **find_similar_faces** searches by face.
5. **forget_faces** erases the faces of the selected images (or of every image), and
   **remove_person** erases a person from the gallery.

## Building a gallery

No gallery is shipped: everyone builds their own, locally, and keeps it to themselves
(see [Never share](#never-share)). A gallery is a folder with one sub-folder per person,
5–20 varied photos each, where the person has the largest face:

    my_gallery/Marie Curie/1.jpg, 2.jpg…
    my_gallery/Victor Hugo/1.jpg…

Make one by hand and run **import_gallery** on it, or select images in Panoptic and run
**add_reference**. For public figures, `tools/build_gallery.py` builds one from public
sources (run it with the Panoptic backend's Python; `--limit 50` for a first try; it
resumes where it stopped):

    python tools/build_gallery.py all --contact <url or email>

Then run **import_gallery** on `gallery/`, and delete `gallery_cache/` (it holds birth
dates from official registers).

- Wikidata (through QLever) lists French politicians and public figures, and the most
  notable worldwide; Commons gives the photos.
- Official French sources (`tools/official_fr.py`): every deputy since 2012 from the
  Assemblée nationale open data, with one official portrait per legislature; senators
  from data.senat.fr, with their portrait; mayors of communes of 30,000+ inhabitants
  from the Répertoire national des élus, matched to Wikidata.
- Party: Wikidata's current membership (a party still in existence, then the preferred
  statement, then the latest), else the Assemblée's party of financial attachment.
- The face of each photo is checked against the person's official or Wikidata
  portrait; photos of other people, and people whose face cannot be confirmed, are
  dropped. `persons.csv`, `manifest.csv` (sources, licences) and `conflicts.csv`
  are written next to the photos.

Wikimedia throttles clients without a contact in their User-Agent: `--contact` adds one.

## Tuning

- `match_threshold` (0.45): raise for fewer false names, lower for more recall.
- `match_margin` (0.05): rejects faces close to two gallery persons (look-alikes).
- `det_size` (640): raise to 1024 for small faces in crowd shots. Faces are read from
  Panoptic's largest stored image variant, not the original file.

## Privacy (GDPR)

Face embeddings used to identify people are biometric data, and a party is a political
opinion: both are special categories of personal data (GDPR Art. 9). Whoever runs the
plugin is the data controller and needs a legal basis, an Art. 9 exception and, in most
cases, a data protection impact assessment. See the French README for a plain-language
checklist.

- Everything runs locally; nothing is sent anywhere. Only the models are downloaded.
- This repository holds no photo, no gallery and no face embedding. The plugin never
  exports embeddings.
- The plugin and the gallery builder write a `.gitignore` (`*`) into every folder
  holding faces, so git skips them even when copied elsewhere.
- Erasure: **forget_faces**, **remove_person**, or delete `plugin_data/faces/`. Deleted
  rows are overwritten (SQLite `secure_delete`).
- The gallery builder only lists adults (born 2007 at the latest). Its cache holds
  personal data from official registers: delete `gallery_cache/` once the gallery is built.

## Never share

Never publish or send, even without the photos:

- a gallery (`gallery/` or your own folder): photos linked to names and parties;
- the builder's cache (`gallery_cache/`): photos and birth dates;
- face embeddings (`plugin_data/faces/faces.db` in the Panoptic project). An embedding
  is 512 numbers computed from a face that recognise the person anywhere: biometric
  data, as sensitive as the photo.

Sharing a Panoptic project? Delete its `plugin_data/faces/` first, or run
**forget_faces** with `everything`. Share the code and the method; let others build
their own gallery.

## Licence

The code is licensed under the GNU GPL v3, like Panoptic (see [LICENSE](LICENSE)).

The InsightFace pretrained weights (`buffalo_l`, `buffalo_s`, `antelopev2`) are licensed
for non-commercial research use only. Gallery photos keep their own licences, listed in
`manifest.csv`.
