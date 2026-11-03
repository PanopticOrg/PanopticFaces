"""Build a reference gallery of public figures from Wikidata and Wikimedia Commons.

Output is the folder layout import_gallery reads — one sub-folder per person, each
photo cropped around that person's face — plus CSV files with sources and licences:

    gallery/<Name>/<n>.jpg
    gallery/manifest.csv     one row per photo: person, QID, Commons file, licence, author
    gallery/persons.csv      one row per person: QID, groups, photos kept
    gallery/conflicts.csv    persons whose faces look alike (same person twice, bad data)

Stages, each resumable and cached under --cache:

    people   Wikidata (through the QLever endpoint) lists the people: French
             parliamentarians and MEPs, French politicians and public figures, and the
             most notable politicians and public figures worldwide. Born 1930 to 2007,
             so no minors (elected officials are adults by law).
    fetch    Commons gives candidate photos: the Wikidata image (P18), then the person's
             Commons category and its portrait / per-year sub-categories. 500 px
             thumbnails are downloaded.
    clean    Faces are detected and embedded with the plugin's engine. The face of the
             Wikidata image is the anchor; a photo is kept when it contains a face close
             to it, so group shots, other people and drawings drop out.

Usage (with the Panoptic backend's Python, which has insightface and onnxruntime):

    python build_gallery.py all --out ~/faces/gallery
    python build_gallery.py clean --out ~/faces/gallery     # re-run one stage

The gallery and the cache hold personal data (photos, birth dates, parties): never
publish or share them. Both folders get a .gitignore that keeps them out of git.

Requests identify themselves as a research tool only. Wikimedia gives low rate limits
to clients without a contact; --contact (an email or URL of your choice) adds one:

    python build_gallery.py all --contact someone@example.org
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import html
import importlib.util
import io
import json
import re
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

USER_AGENT = 'FacesGallery/0.1 (research tool)'    # --contact adds a contact to it
SPARQL_URL = 'https://qlever.dev/api/wikidata'
COMMONS_API = 'https://commons.wikimedia.org/w/api.php'

THUMB_WIDTH = 500            # a standard Commons thumbnail step
MAX_CANDIDATES = 20          # photos downloaded per person
PER_SUBCAT = 4               # photos taken from each sub-category, for variety
MAX_YEARS = 6                # per-year sub-categories visited, spread over the career
MIN_MAYOR_POPULATION = 30000 # mayors of communes at least this large are added
MAX_KEPT = 10                # reference photos kept per person
DOWNLOAD_WORKERS = 3

MIN_DET_SCORE = 0.7
MIN_FACE_PX = 40             # shortest face side in a 500 px thumbnail
SAME_PERSON = 0.45           # ArcFace cosine to the anchor for a photo to be kept
DUPLICATE = 0.92             # near-identical faces (same photo twice) are kept once
CONFLICT = 0.6               # person centroids this close are cross-checked, then reported
WEAK_SET = 3                 # a look-alike with this many photos or fewer is dropped
CROP_MARGIN = 2.0            # crop side, as a multiple of the face box

OCCUPATIONS = {
    'Q33999': 'actor', 'Q177220': 'singer', 'Q1930187': 'journalist',
    'Q947873': 'television presenter', 'Q937857': 'football player', 'Q2252262': 'rapper',
    'Q639669': 'musician', 'Q43845': 'businessperson', 'Q2526255': 'film director',
    'Q245068': 'comedian', 'Q17125263': 'YouTuber', 'Q2066131': 'athlete',
    'Q3665646': 'basketball player', 'Q10833314': 'tennis player',
}

PREFIXES = '''
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
'''
# Every query selects ?p plus these, so one parser reads them all.
COLUMNS = '''
  OPTIONAL { ?p rdfs:label ?fr FILTER(LANG(?fr) = "fr") }
  OPTIONAL { ?p rdfs:label ?en FILTER(LANG(?en) = "en") }
  OPTIONAL { ?p rdfs:label ?mul FILTER(LANG(?mul) = "mul") }
  OPTIONAL { ?p wdt:P18 ?img }
  OPTIONAL { ?p wdt:P373 ?cat }
  OPTIONAL { ?p wikibase:sitelinks ?sl }
  OPTIONAL { ?p wdt:P569 ?birth }
  FILTER(BOUND(?img) || BOUND(?cat))
'''
SELECT = 'SELECT DISTINCT ?p ?fr ?en ?mul ?img ?cat ?sl ?birth WHERE {'
CELEBS = ' '.join(f'wd:{q}' for q in OCCUPATIONS)

GROUPS = {
    'fr_parliament': f'''{SELECT}
        VALUES ?office {{ wd:Q3044918 wd:Q14828018 }}
        ?p p:P39 ?st . ?st ps:P39 ?office .
        OPTIONAL {{ ?st pq:P580 ?s }} OPTIONAL {{ ?st pq:P582 ?e }}
        FILTER((BOUND(?s) && YEAR(?s) >= 2012) || (BOUND(?e) && YEAR(?e) >= 2012))
        {COLUMNS} }}''',
    'fr_mep': f'''{SELECT}
        ?p wdt:P27 wd:Q142 ; p:P39 ?st . ?st ps:P39 wd:Q27169 ; pq:P580 ?s .
        FILTER(YEAR(?s) >= 2009)
        {COLUMNS} }}''',
    'fr_politician': f'''{SELECT}
        ?p wdt:P106 wd:Q82955 ; wdt:P27 wd:Q142 ; wdt:P569 ?b ; wikibase:sitelinks ?n .
        FILTER(?n >= 10 && YEAR(?b) >= 1930 && YEAR(?b) <= 2007)
        {COLUMNS} }}''',
    'fr_public': f'''{SELECT}
        VALUES ?occ {{ {CELEBS} }}
        ?p wdt:P106 ?occ ; wdt:P27 wd:Q142 ; wdt:P569 ?b ; wikibase:sitelinks ?n .
        FILTER(?n >= 20 && YEAR(?b) >= 1930 && YEAR(?b) <= 2007)
        {COLUMNS} }}''',
    'world_politician': f'''{SELECT}
        ?p wdt:P106 wd:Q82955 ; wdt:P569 ?b ; wikibase:sitelinks ?n .
        FILTER(?n >= 60 && YEAR(?b) >= 1930 && YEAR(?b) <= 2007)
        {COLUMNS} }}''',
    'world_public': f'''{SELECT}
        VALUES ?occ {{ {CELEBS} }}
        ?p wdt:P106 ?occ ; wdt:P569 ?b ; wikibase:sitelinks ?n .
        FILTER(?n >= 80 && YEAR(?b) >= 1930 && YEAR(?b) <= 2007)
        {COLUMNS} }}''',
}

# Commons files and sub-categories that are not photos of the person's face.
BAD_FILE = re.compile(
    r'signature|autograph|logo|coat.of.arms|blason|armoiries|\bmap\b|carte|flag|drapeau|'
    r'caricature|cartoon|drawing|dessin|painting|peinture|mural|graffiti|statue|\bbust\b|'
    r'buste|\bwax\b|\bcire\b|tussaud|gr[ée]vin|montage|collage|poster|affiche|stamp|'
    r'timbre|banknote|\bcoin\b|grave|tomb|plaque|sticker|\bmask', re.I)
BAD_SUBCAT = re.compile(
    r'caricature|drawing|dessin|painting|photomontage|signature|\bmaps?\b|protest|'
    r'demonstration|manifestation|cartoon|statue|\bwax\b|\bart\b|mural|graffiti|poster|'
    r'\bcoins?\b|stamps?\b|popularity|related to|logo|audio|video|document|by location|'
    r'with people|shaking hands|masks|family|relatives|residence|house|tomb|grave|'
    r'in fiction|in popular culture|sculpture', re.I)
# A file title naming several people: its largest face may be any of them.
SEVERAL_PEOPLE = re.compile(r'\s&\s|\b(and|et|with|avec|und|mit|con|e|i)\b|\s\+\s', re.I)
GOOD_SUBCAT = re.compile(r'portrait|quality images|official|\bin \d{4}\b|\bby year\b', re.I)


def ignore_folder(folder: Path) -> None:
    """Keep ``folder`` out of any git repository: it holds personal data."""
    folder.mkdir(parents=True, exist_ok=True)
    marker = folder / '.gitignore'
    if not marker.exists():
        marker.write_text('# Personal data (photos of people): never publish\n*\n')


def log(message: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def http_get(url: str, data: bytes | None = None, accept: str = '*/*', retries: int = 5) -> bytes:
    """GET (or POST with ``data``), retrying on throttling and transient errors."""
    for attempt in range(retries):
        req = urllib.request.Request(url, data=data, headers={'User-Agent': USER_AGENT,
                                                              'Accept': accept})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < retries - 1:
                wait = float(e.headers.get('Retry-After') or 2 ** (attempt + 1))
                time.sleep(min(wait, 120))
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            if attempt < retries - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise
    raise RuntimeError('unreachable')


_api_lock = threading.Lock()


def commons(**params) -> dict:
    """One Commons API call. Calls are made one at a time, as the API etiquette asks."""
    params = {'action': 'query', 'format': 'json', 'formatversion': '2', 'maxlag': '5', **params}
    url = f'{COMMONS_API}?{urllib.parse.urlencode(params)}'
    with _api_lock:
        for _ in range(5):
            data = json.loads(http_get(url, accept='application/json'))
            if data.get('error', {}).get('code') == 'maxlag':
                time.sleep(5)
                continue
            return data
    raise RuntimeError(f'Commons stayed lagged for {url}')


# ---------------------------------------------------------------------------
# Stage 1 — people
# ---------------------------------------------------------------------------

def _display_name(fr: str | None, en: str | None, mul: str | None) -> str | None:
    """The French label, unless it is a single word ('François' for the pope) and the
    English one is not ('Pope Francis') — a tag must say who it is on its own.

    Wikidata now stores a name shared by all languages once, as the 'mul' label, and
    drops the identical fr / en ones — Emmanuel Macron has only 'mul'."""
    fr, en = fr or mul, en or mul
    if fr and ' ' not in fr.strip() and en and ' ' in en.strip():
        return en
    return fr or en


def _sparql(query: str) -> list[dict]:
    raw = http_get(SPARQL_URL, data=urllib.parse.urlencode({'query': PREFIXES + query}).encode(),
                   accept='application/sparql-results+json')
    return json.loads(raw)['results']['bindings']


def _new_person(key: str) -> dict:
    return {'qid': key, 'name': None, 'fr': None, 'en': None, 'mul': None, 'groups': [],
            'images': [], 'category': None, 'sitelinks': 0, 'birth': None,
            'party': None, 'group': None, 'mandates': [], 'official': []}


def _add_rows(people: dict[str, dict], rows: list[dict], group: str) -> None:
    for row in rows:
        qid = row['p']['value'].rsplit('/', 1)[-1]
        person = people.setdefault(qid, _new_person(qid))
        for lang in ('fr', 'en', 'mul'):
            if lang in row and not person[lang]:
                person[lang] = row[lang]['value']
        if group not in person['groups']:
            person['groups'].append(group)
        if 'img' in row:
            file = urllib.parse.unquote(row['img']['value'].rsplit('/', 1)[-1]).replace('_', ' ')
            if file not in person['images']:
                person['images'].append(file)
        if 'cat' in row:
            person['category'] = row['cat']['value']
        if 'sl' in row:
            person['sitelinks'] = max(person['sitelinks'], int(row['sl']['value']))
        if 'birth' in row and not person['birth']:
            person['birth'] = row['birth']['value'][:4]


def _wikidata_details(people: dict[str, dict], qids: list[str], group: str) -> None:
    """Labels, images and category of ``qids`` — with or without a photo on Wikidata."""
    columns = COLUMNS.replace('FILTER(BOUND(?img) || BOUND(?cat))', '')
    for i in range(0, len(qids), 300):
        values = ' '.join(f'wd:{q}' for q in qids[i:i + 300])
        _add_rows(people, _sparql(f'{SELECT} VALUES ?p {{ {values} }} {columns} }}'), group)


def _load_official():
    path = Path(__file__).resolve().with_name('official_fr.py')
    spec = importlib.util.spec_from_file_location('official_fr', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _norm(text: str) -> str:
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z]+', ' ', text).strip()


def _add_official(cache: Path, people: dict[str, dict]) -> None:
    """Deputies, senators and big-city mayors from the official French sources."""
    off = _load_official()
    fetch = lambda url: http_get(url)

    # Assemblée nationale: every deputy since 2012, linked to Wikidata by its AN id (P4123).
    deputies = off.deputies(cache, fetch)
    an_ids = {r['id']['value']: r['p']['value'].rsplit('/', 1)[-1]
              for r in _sparql('SELECT ?p ?id WHERE { ?p wdt:P4123 ?id }')}
    _wikidata_details(people, sorted({an_ids[n] for n in deputies if n in an_ids} - set(people)),
                      'fr_parliament')
    linked = 0
    for num, d in deputies.items():
        key = an_ids.get(num) or f'AN{num}'
        linked += key in an_ids.values()
        person = people.setdefault(key, _new_person(key))
        person['an_name'] = d['name']
        if not person['birth'] and d['birth']:
            person['birth'] = d['birth'][:4]
        if 'fr_parliament' not in person['groups']:
            person['groups'].append('fr_parliament')
        # PARPOL is the party of *financial attachment*: for many deputies a coalition
        # ('Ensemble ! (majorité présidentielle)'), so it only fills in for Wikidata.
        party = d['party'] if d['party'] and not d['party'].startswith('Non ') else None
        person['an_party'] = party
        person['group'] = d['group'] or person['group']
        person['mandates'].append('Député ' + ', '.join(f'{leg}e' for leg in d['legislatures']))
        person['official'] += d['photos']
    log(f'  Assemblée nationale: {len(deputies)} deputies since 2012, {linked} linked to Wikidata')

    # Sénat: sitting senators, and former ones already in the gallery. The portrait URL
    # comes from the senat.fr id (P1808), whose slug ends with the matricule.
    senators = off.senators(cache, fetch)
    rows = _sparql('SELECT ?p ?id WHERE { ?p wdt:P1808 ?id FILTER(STRSTARTS(?id, "senateur/")) }')
    by_matricule = {}
    for r in rows:
        slug = r['id']['value'].split('/', 1)[1]
        m = re.search(r'(\d{5}[a-z])$', slug)
        if m:
            by_matricule[m.group(1)] = (r['p']['value'].rsplit('/', 1)[-1], slug)
    chosen = {mat: by_matricule[mat] for mat, s in senators.items()
              if mat in by_matricule and (s['active'] or by_matricule[mat][0] in people)}
    _wikidata_details(people, sorted({q for q, _ in chosen.values()} - set(people)), 'fr_parliament')
    for mat, (qid, slug) in chosen.items():
        person = people.setdefault(qid, _new_person(qid))
        s = senators[mat]
        if 'fr_parliament' not in person['groups']:
            person['groups'].append('fr_parliament')
        person['group'] = person['group'] or s['group']
        person['mandates'].append('Sénateur' + (' (en exercice)' if s['active'] else ''))
        person['official'].append(off.senator_photo(slug))
    active = sum(s['active'] for s in senators.values())
    log(f'  Sénat: {len(chosen)} senators with a portrait '
        f'({sum(senators[m]["active"] for m in chosen)}/{active} sitting ones)')

    # Mayors of communes of at least MIN_MAYOR_POPULATION inhabitants: names from the
    # RNE, matched to Wikidata by birth date and name for their photos.
    communes = _sparql(f'''SELECT ?insee (MAX(?pop) AS ?population) WHERE {{
        ?c wdt:P31 wd:Q484170 ; wdt:P374 ?insee ; wdt:P1082 ?pop }}
        GROUP BY ?insee HAVING (MAX(?pop) >= {MIN_MAYOR_POPULATION})''')
    mayors = off.mayors(cache, fetch, {r['insee']['value'] for r in communes})
    dates = sorted({m['birth'] for m in mayors if m['birth']})
    values = ' '.join(f'"{d}T00:00:00Z"^^xsd:dateTime' for d in dates)
    rows = _sparql(f'''SELECT ?p ?b ?fr ?mul WHERE {{ VALUES ?b {{ {values} }}
        ?p wdt:P569 ?b ; wdt:P31 wd:Q5 .
        OPTIONAL {{ ?p rdfs:label ?fr FILTER(LANG(?fr) = "fr") }}
        OPTIONAL {{ ?p rdfs:label ?mul FILTER(LANG(?mul) = "mul") }} }}''')
    by_birth: dict[str, list[tuple[str, str]]] = {}
    for r in rows:
        label = (r.get('fr') or r.get('mul') or {}).get('value')
        if label:
            by_birth.setdefault(r['b']['value'][:10], []).append((r['p']['value'].rsplit('/', 1)[-1], label))
    matched: dict[str, dict] = {}
    for m in mayors:
        last, first = _norm(m['last']), _norm(m['first']).split(' ')[0]
        for qid, label in by_birth.get(m['birth'] or '', []):
            if last in _norm(label) and first in _norm(label).split(' '):
                matched[qid] = m
                break
    _wikidata_details(people, sorted(set(matched) - set(people)), 'fr_mayor')
    for qid, m in matched.items():
        person = people[qid]
        if 'fr_mayor' not in person['groups']:
            person['groups'].append('fr_mayor')
        person['mandates'].append(f"Maire de {m['commune']}")
    log(f'  RNE: {len(mayors)} mayors of communes >= {MIN_MAYOR_POPULATION} inhabitants, '
        f'{len(matched)} found on Wikidata')


def _add_parties(people: dict[str, dict]) -> None:
    """Current party: Wikidata P102 without end date — a party still in existence first,
    then the statement ranked 'preferred', then the most recent start, since old
    memberships are often left open. The
    Assemblée's party of financial attachment fills in when Wikidata has none."""
    qids = [q for q in people if q.startswith('Q')]
    for i in range(0, len(qids), 400):
        values = ' '.join(f'wd:{q}' for q in qids[i:i + 400])
        rows = _sparql(f'''SELECT ?p ?fr ?mul ?start ?rank ?gone WHERE {{ VALUES ?p {{ {values} }}
            ?p p:P102 ?st . ?st ps:P102 ?party ; wikibase:rank ?rank .
            FILTER NOT EXISTS {{ ?st pq:P582 ?end }}
            OPTIONAL {{ ?party wdt:P576 ?gone }}
            OPTIONAL {{ ?st pq:P580 ?start }}
            OPTIONAL {{ ?party rdfs:label ?fr FILTER(LANG(?fr) = "fr") }}
            OPTIONAL {{ ?party rdfs:label ?mul FILTER(LANG(?mul) = "mul") }} }}''')
        best: dict[str, tuple[tuple, str]] = {}
        for r in rows:
            label = (r.get('fr') or r.get('mul') or {}).get('value')
            # Deprecated statements are dropped here: QLever's IRI != filter matches nothing.
            if not label or r['rank']['value'].endswith('DeprecatedRank'):
                continue
            qid = r['p']['value'].rsplit('/', 1)[-1]
            # A party that no longer exists (RPR, UMP…) loses to any current one: old
            # memberships are often left without an end date.
            key = ('gone' not in r, r['rank']['value'].endswith('PreferredRank'), r.get('start', {}).get('value', ''))
            if qid not in best or key > best[qid][0]:
                best[qid] = (key, label)
        for qid, (_, label) in best.items():
            people[qid]['party'] = label
    for person in people.values():
        person['party'] = person['party'] or person.get('an_party')
    # One spelling per party: 'La France Insoumise' (Assemblée) and 'La France insoumise'
    # (Wikidata) would otherwise be two tags.
    spellings = collections.Counter(p['party'] for p in people.values() if p['party'])
    canonical = {}
    for label, _ in spellings.most_common():
        canonical.setdefault(label.casefold(), label)
    for person in people.values():
        if person['party']:
            person['party'] = canonical[person['party'].casefold()]
    log(f'  parties: {sum(1 for p in people.values() if p["party"])} people with a party')


def stage_people(cache: Path) -> list[dict]:
    people: dict[str, dict] = {}
    for group, query in GROUPS.items():
        log(f'Wikidata: {group}')
        rows = _sparql(query)
        _add_rows(people, rows, group)
        log(f'  {len(rows)} rows, {len(people)} people so far')
        time.sleep(1)
    log('Official French sources')
    _add_official(cache, people)
    _add_parties(people)

    for person in people.values():
        person['name'] = (_display_name(person['fr'], person['en'], person['mul'])
                          or person.get('an_name'))
    unnamed = [p['qid'] for p in people.values() if not p['name']]
    if unnamed:
        log(f'  {len(unnamed)} people without any name skipped: {unnamed[:10]}')
    result = [p for p in people.values() if p['name']]
    result.sort(key=lambda p: -p['sitelinks'])
    (cache / 'people.json').write_text(json.dumps(result, ensure_ascii=False, indent=1))
    log(f'{len(result)} people')
    return result


# ---------------------------------------------------------------------------
# Stage 2 — fetch candidate photos
# ---------------------------------------------------------------------------

def _files_info(titles: list[str]) -> dict[str, dict]:
    """Thumbnail URL, licence and author of Commons files, 50 per call, by requested title."""
    out: dict[str, dict] = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        data = commons(titles='|'.join(chunk), prop='imageinfo',
                       iiprop='url|mime|extmetadata', iiurlwidth=THUMB_WIDTH,
                       iiextmetadatafilter='LicenseShortName|Artist')
        # The API answers with normalised titles; map them back to the requested ones.
        renamed = {n['to']: n['from'] for n in data.get('query', {}).get('normalized', [])}
        for info in _pages_info(data):
            out[renamed.get(info['title'], info['title'])] = info
    return out


def _category_files(category: str, limit: int) -> list[dict]:
    data = commons(generator='categorymembers', gcmtitle=f'Category:{category}', gcmtype='file',
                   gcmlimit=str(min(limit * 3, 50)), prop='imageinfo',
                   iiprop='url|mime|extmetadata', iiurlwidth=THUMB_WIDTH,
                   iiextmetadatafilter='LicenseShortName|Artist')
    return [f for f in _pages_info(data) if not BAD_FILE.search(f['title'])][:limit]


def _subcategories(category: str) -> list[str]:
    data = commons(list='categorymembers', cmtitle=f'Category:{category}', cmtype='subcat',
                   cmlimit='200')
    return [m['title'].split(':', 1)[1] for m in data.get('query', {}).get('categorymembers', [])]


def _pages_info(data: dict) -> list[dict]:
    out = []
    for page in data.get('query', {}).get('pages', []):
        info = (page.get('imageinfo') or [None])[0]
        if not info or info.get('mime') not in ('image/jpeg', 'image/png') or 'thumburl' not in info:
            continue
        meta = info.get('extmetadata')
        meta = meta if isinstance(meta, dict) else {}        # an empty list when there is none
        field = lambda key: str((meta.get(key) or {}).get('value', '') if isinstance(meta.get(key), dict) else '')
        artist = re.sub(r'<[^>]+>', '', html.unescape(field('Artist'))).strip()
        out.append({'title': page['title'], 'thumb': info['thumburl'],
                    'page': info.get('descriptionurl', ''),
                    'license': field('LicenseShortName'),
                    'artist': artist[:200]})
    return out


def _year(category: str) -> int:
    years = re.findall(r'\b(\d{4})\b', category)
    return int(years[-1]) if years else 0


def _mentions(title: str, name: str) -> bool:
    """True when a file name contains a distinctive word of the person's name."""
    fold = lambda t: unicodedata.normalize('NFKD', t).encode('ascii', 'ignore').decode().lower()
    words = [w for w in re.split(r'[\s\-.]+', fold(name)) if len(w) >= 4]
    return any(w in fold(title) for w in words)


def _spread(items: list, k: int) -> list:
    """Up to ``k`` items evenly spaced over ``items``."""
    if len(items) <= k:
        return items
    return [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def _candidates(person: dict, wikidata_images: dict[str, dict]) -> list[dict]:
    """Wikidata image(s) first (they are the anchor), then Commons category photos."""
    found: dict[str, dict] = {}

    def add(files: list[dict], anchor: bool = False) -> None:
        for f in files:
            if len(found) >= MAX_CANDIDATES:
                return
            if f['title'] not in found:
                found[f['title']] = {**f, 'anchor': anchor}

    add([wikidata_images[f'File:{t}'] for t in person['images'] if f'File:{t}' in wikidata_images],
        anchor=True)

    category = person['category']
    if category and len(found) < MAX_CANDIDATES:
        subcats = [c for c in _subcategories(category) if not BAD_SUBCAT.search(c)]
        good = [c for c in subcats if GOOD_SUBCAT.search(c)]
        # "X by year" holds "X in 2017", "X in 2018"… — the per-year photos spread the ages.
        years = []
        for c in [c for c in good if re.search(r'by year', c, re.I)]:
            years += [y for y in _subcategories(c) if not BAD_SUBCAT.search(y)]
        years = _spread(sorted(years, key=_year), MAX_YEARS)[::-1]      # newest first
        portraits = [c for c in good if not re.search(r'by year', c, re.I)]
        portraits.sort(key=lambda c: (not re.search(r'portrait|official', c, re.I), -_year(c)))
        # Portraits, then years spread over the career, then the category's own files —
        # those are often maps, houses or events, so only the ones naming the person.
        for sub in portraits + years:
            if len(found) >= MAX_CANDIDATES:
                break
            add(_category_files(sub, PER_SUBCAT))
        if len(found) < MAX_CANDIDATES:
            add([f for f in _category_files(category, PER_SUBCAT * 3)
                 if _mentions(f['title'], person['name'])])
    return list(found.values())


def _download(item: tuple[Path, str]) -> bool:
    path, url = item
    if path.exists():
        return True
    try:
        data = http_get(url, accept='image/*')
    except Exception as e:
        log(f'  download failed: {url[:100]} ({e})')
        return False
    tmp = path.with_suffix('.part')
    tmp.write_bytes(data)
    tmp.rename(path)
    return True


def stage_fetch(cache: Path, people: list[dict]) -> None:
    meta_dir, img_dir = cache / 'meta', cache / 'img'
    meta_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in people if not (meta_dir / f'{p["qid"]}.json').exists() and p['images']]
    log(f'fetch: Wikidata images of {len(todo)} people, 50 per call')
    wikidata_images = _files_info([f'File:{t}' for p in todo for t in p['images']])

    pool = ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS)
    pending = []
    started = time.time()
    for n, person in enumerate(people, 1):
        meta_file = meta_dir / f'{person["qid"]}.json'
        if meta_file.exists():
            files = json.loads(meta_file.read_text())
        else:
            try:
                files = _candidates(person, wikidata_images)
            except Exception as e:
                log(f'  {person["name"]} ({person["qid"]}): Commons failed ({e})')
                continue
            meta_file.write_text(json.dumps(files, ensure_ascii=False))
        # Official portraits come first: they are the most reliable anchor.
        known = {f['title'] for f in files}
        official = [{**o, 'anchor': True} for o in person.get('official', []) if o['title'] not in known]
        if official:
            files = official + files
            meta_file.write_text(json.dumps(files, ensure_ascii=False))
        folder = img_dir / person['qid']
        folder.mkdir(parents=True, exist_ok=True)
        for f in files:
            name = hashlib.md5(f['title'].encode()).hexdigest()[:16] + '.jpg'
            pending.append(pool.submit(_download, (folder / name, f['thumb'])))
        if n % 50 == 0:
            done = sum(fut.done() for fut in pending)
            rate = n / (time.time() - started)
            log(f'fetch: {n}/{len(people)} people, {done}/{len(pending)} photos, '
                f'~{(len(people) - n) / rate / 60:.0f} min left')
    pool.shutdown(wait=True)
    ok = sum(fut.result() for fut in pending)
    log(f'fetch done: {ok}/{len(pending)} photos')


# ---------------------------------------------------------------------------
# Stage 3 — clean: keep the photos where the person's face is
# ---------------------------------------------------------------------------

def _load_engine_module():
    """Import the plugin's engine.py on its own — no Panoptic needed."""
    path = Path(__file__).resolve().parents[1] / 'faces_plugin' / 'engine.py'
    spec = importlib.util.spec_from_file_location('faces_engine', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _folder_name(name: str) -> str:
    return re.sub(r'[/\\:*?"<>|]', ' ', name).strip()


def _select(faces_per_photo, anchor_idx, np, titles):
    """Pick, per photo, the face of the person; return [(photo index, face index, cos)]."""
    def support(i, f):
        """In how many other photos this face is seen again."""
        return sum(1 for j, other in enumerate(faces_per_photo) if j != i and other and
                   max(float(f['embedding'] @ g['embedding']) for g in other) >= SAME_PERSON)

    # Anchor: the face of the Wikidata image — its largest face only when it clearly
    # dominates; in a group photo, the face the other photos confirm. Without one, the
    # face most often seen again (the person is the one face common to the category).
    anchor = None
    for i in anchor_idx:
        faces = sorted(faces_per_photo[i], key=lambda f: -f['area'])
        if not faces:
            continue
        # A title naming several people proves nothing about the one face found: the
        # person may be the one too small or cut off to be detected.
        several = SEVERAL_PEOPLE.search(titles[i].rsplit('.', 1)[0])
        dominant = len(faces) == 1 or faces[0]['area'] >= 2 * faces[1]['area']
        if dominant and not several:
            anchor = faces[0]['embedding']
        else:
            scored = [(support(i, f), f) for f in faces]
            best_support, best = max(scored, key=lambda t: t[0])
            if best_support:
                anchor = best['embedding']
        if anchor is not None:
            break
    if anchor is None:
        best, best_support = None, 0
        for i, faces in enumerate(faces_per_photo):
            for f in faces:
                s = support(i, f)
                if s > best_support:
                    best, best_support = f['embedding'], s
        if best is None:
            # A single photo with a single face is accepted as is.
            single = [fs for fs in faces_per_photo if len(fs) == 1]
            if (len(single) == 1 and len(faces_per_photo) == 1
                    and not SEVERAL_PEOPLE.search(titles[0].rsplit('.', 1)[0])):
                best = single[0][0]['embedding']
            else:
                return []
        anchor = best

    def pick(reference):
        chosen = []
        for i, faces in enumerate(faces_per_photo):
            if not faces:
                continue
            sims = [float(f['embedding'] @ reference) for f in faces]
            k = int(np.argmax(sims))
            if sims[k] >= SAME_PERSON:
                chosen.append((i, k, sims[k]))
        return chosen

    chosen = pick(anchor)
    if len(chosen) >= 3:
        # Second pass against the centroid, which is steadier than one photo.
        centroid = np.mean([faces_per_photo[i][k]['embedding'] for i, k, _ in chosen], axis=0)
        chosen = pick(centroid / np.linalg.norm(centroid))
    return chosen


def stage_clean(cache: Path, out: Path, people: list[dict], insightface_root: Path) -> None:
    import numpy as np
    from PIL import Image

    engine_mod = _load_engine_module()
    engine = engine_mod.FaceEngine(insightface_root, 'buffalo_l', 'auto', 640)
    log(f'face engine on {engine.backend}')

    out.mkdir(parents=True, exist_ok=True)
    used_names: dict[str, str] = {}
    # name -> person row, and its references [(embedding, JPEG bytes, manifest row)];
    # written only after the cross-check between people.
    kept: dict[str, dict] = {}
    started = time.time()

    for n, person in enumerate(people, 1):
        meta_file = cache / 'meta' / f'{person["qid"]}.json'
        if not meta_file.exists():
            continue
        files = json.loads(meta_file.read_text())
        photos, metas = [], []
        for f in files:
            path = cache / 'img' / person['qid'] / (hashlib.md5(f['title'].encode()).hexdigest()[:16] + '.jpg')
            if not path.exists():
                continue
            try:
                photos.append(np.asarray(Image.open(path).convert('RGB')))
                metas.append(f)
            except Exception:
                continue
        if not photos:
            continue

        detected = engine.analyse_batch(photos, MIN_DET_SCORE, 0)
        faces_per_photo = []
        for img, faces in zip(photos, detected):
            h, w = img.shape[:2]
            usable = []
            for f in faces:
                x1, y1, x2, y2 = f['box']
                fw, fh = (x2 - x1) * w, (y2 - y1) * h
                if min(fw, fh) >= MIN_FACE_PX:
                    usable.append({**f, 'area': fw * fh})
            faces_per_photo.append(usable)

        chosen = _select(faces_per_photo, [i for i, m in enumerate(metas) if m['anchor']], np,
                         [m['title'] for m in metas])
        # Best matches first, near-duplicates dropped, at most MAX_KEPT.
        chosen.sort(key=lambda c: -c[2])
        selected, embs = [], []
        for i, k, sim in chosen:
            e = faces_per_photo[i][k]['embedding']
            if any(float(e @ o) >= DUPLICATE for o in embs):
                continue
            selected.append((i, k, sim))
            embs.append(e)
            if len(selected) >= MAX_KEPT:
                break
        if not selected:
            continue

        name = _folder_name(person['name'])
        if name in used_names and used_names[name] != person['qid']:
            name = f'{name} ({person["qid"]})'
        used_names[name] = person['qid']

        refs = []
        for i, k, sim in selected:
            img = photos[i]
            h, w = img.shape[:2]
            x1, y1, x2, y2 = faces_per_photo[i][k]['box']
            cx, cy = (x1 + x2) / 2 * w, (y1 + y2) / 2 * h
            half = max((x2 - x1) * w, (y2 - y1) * h) * CROP_MARGIN / 2
            box = (int(max(cx - half, 0)), int(max(cy - half, 0)),
                   int(min(cx + half, w)), int(min(cy + half, h)))
            buf = io.BytesIO()
            Image.fromarray(img).crop(box).save(buf, format='JPEG', quality=92)
            m = metas[i]
            refs.append((faces_per_photo[i][k]['embedding'], buf.getvalue(),
                         {'person': name, 'qid': person['qid'], 'file': '',
                          'similarity': round(sim, 3), 'source': m['title'],
                          'page': m['page'], 'license': m['license'], 'author': m['artist']}))
        kept[name] = {'row': {'person': name, 'qid': person['qid'],
                              'groups': ' '.join(person['groups']),
                              'party': person.get('party') or '',
                              'group': person.get('group') or '',
                              'mandates': ' | '.join(person.get('mandates', [])),
                              'sitelinks': person['sitelinks'], 'birth': person['birth'],
                              'photos': 0, 'candidates': len(photos)},
                      'refs': refs}
        if n % 100 == 0:
            rate = n / (time.time() - started)
            log(f'clean: {n}/{len(people)} people, {len(kept)} kept, '
                f'~{(len(people) - n) / rate / 60:.0f} min left')
    engine.close()

    dropped = _cross_check(kept, np)
    _write_gallery(out, kept, np)
    log(f'cross-check: {dropped} photos dropped as another person\'s face')


def _centroid(embs, np):
    c = np.mean(embs, axis=0)
    return c / np.linalg.norm(c)


def _cross_check(kept: dict[str, dict], np) -> int:
    """Drop references that look more like another person than like their own set.

    A group photo can lend one person's face to another (a deputy's Wikidata image
    showing a colleague). For each pair of people whose faces are close, a reference
    goes when it is nearer the other person's centroid than the rest of its own set —
    or, for a person with a single reference, when it matches the other at all closely.
    Twins keep their own sets: each photo stays nearer its own person.
    """
    names = list(kept)
    if len(names) < 2:
        return 0
    mat = np.stack([_centroid([e for e, _, _ in kept[n]['refs']], np) for n in names])
    sims = mat @ mat.T
    dropped = 0
    for a, b in zip(*np.nonzero(np.triu(sims, 1) >= CONFLICT)):
        for me, other in ((names[a], names[b]), (names[b], names[a])):
            refs, other_refs = kept[me]['refs'], kept[other]['refs']
            if not refs or not other_refs:
                continue
            other_c = _centroid([e for e, _, _ in other_refs], np)
            keep = []
            for i, ref in enumerate(refs):
                rest = [e for j, (e, _, _) in enumerate(refs) if j != i]
                to_other = float(ref[0] @ other_c)
                if rest:
                    wrong = to_other > float(ref[0] @ _centroid(rest, np))
                else:
                    wrong = to_other >= CONFLICT
                if not wrong:
                    keep.append(ref)
            dropped += len(refs) - len(keep)
            kept[me]['refs'] = keep
    for name in [n for n in names if not kept[n]['refs']]:
        del kept[name]

    # A whole set can be someone else's face — a few photos of one event, all showing a
    # better-known person next to them — and then it agrees with itself. When two people
    # still look alike, the one with thin evidence (WEAK_SET photos or fewer; then the
    # less notable) goes. Twins with fuller sets are both kept.
    names = list(kept)
    mat = np.stack([_centroid([e for e, _, _ in kept[n]['refs']], np) for n in names])
    sims = mat @ mat.T
    gone: set[str] = set()
    for a, b in sorted(zip(*np.nonzero(np.triu(sims, 1) >= CONFLICT)), key=lambda ab: -sims[ab]):
        pa, pb = names[a], names[b]
        if pa in gone or pb in gone:
            continue
        weak = min((pa, pb), key=lambda n: (len(kept[n]['refs']), kept[n]['row']['sitelinks']))
        if len(kept[weak]['refs']) <= WEAK_SET:
            log(f'  dropped {weak}: looks like {pb if weak == pa else pa} ({sims[a, b]:.2f})')
            dropped += len(kept[weak]['refs'])
            gone.add(weak)
    for name in gone:
        del kept[name]
    return dropped


def _write_gallery(out: Path, kept: dict[str, dict], np) -> None:
    # Remove the folders of the previous build, which may name people dropped since.
    previous = out / 'persons.csv'
    if previous.exists():
        for row in csv.DictReader(previous.open(encoding='utf-8')):
            folder = out / row['person']
            if folder.is_dir():
                for f in folder.glob('*.jpg'):
                    f.unlink()
                if not any(folder.iterdir()):
                    folder.rmdir()

    manifest, persons_rows, centroids = [], [], []
    for name, person in kept.items():
        folder = out / name
        folder.mkdir(parents=True, exist_ok=True)
        for j, (_, data, row) in enumerate(person['refs'], 1):
            (folder / f'{j}.jpg').write_bytes(data)
            manifest.append({**row, 'file': f'{name}/{j}.jpg'})
        persons_rows.append({**person['row'], 'photos': len(person['refs'])})
        centroids.append((name, person['row']['qid'],
                          _centroid([e for e, _, _ in person['refs']], np)))
    _write_csv(out / 'manifest.csv', manifest)
    _write_csv(out / 'persons.csv', persons_rows)

    # What is still this close after the cross-check: twins, or worth a look.
    conflicts = []
    if centroids:
        mat = np.stack([c for _, _, c in centroids])
        sims = mat @ mat.T
        for a, b in zip(*np.nonzero(np.triu(sims, 1) >= CONFLICT)):
            conflicts.append({'person_a': centroids[a][0], 'qid_a': centroids[a][1],
                              'person_b': centroids[b][0], 'qid_b': centroids[b][1],
                              'similarity': round(float(sims[a, b]), 3)})
    _write_csv(out / 'conflicts.csv', conflicts)
    log(f'clean done: {len(persons_rows)} people, {len(manifest)} photos, '
        f'{len(conflicts)} look-alike pairs left to check')


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        if not rows:
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------

def main() -> None:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['people', 'fetch', 'clean', 'all'])
    parser.add_argument('--out', type=Path, default=here / 'gallery')
    parser.add_argument('--cache', type=Path, default=here / 'gallery_cache')
    parser.add_argument('--insightface-root', type=Path, default=None,
                        help='folder holding models/buffalo_l (default: <cache>/insightface)')
    parser.add_argument('--limit', type=int, default=0, help='only the N most notable people')
    parser.add_argument('--contact', default='',
                        help='email or URL added to the User-Agent; Wikimedia throttles clients without one')
    args = parser.parse_args()

    global USER_AGENT
    if args.contact:
        USER_AGENT = f'FacesGallery/0.1 (research tool; {args.contact})'

    args.cache.mkdir(parents=True, exist_ok=True)
    ignore_folder(args.cache)
    if args.stage in ('clean', 'all'):
        ignore_folder(args.out)
    people_file = args.cache / 'people.json'
    if args.stage in ('people', 'all') or not people_file.exists():
        people = stage_people(args.cache)
    else:
        people = json.loads(people_file.read_text())
    if args.limit:
        people = people[:args.limit]
    if args.stage in ('fetch', 'all'):
        stage_fetch(args.cache, people)
    if args.stage in ('clean', 'all'):
        stage_clean(args.cache, args.out, people, args.insightface_root or args.cache / 'insightface')


if __name__ == '__main__':
    sys.exit(main())
