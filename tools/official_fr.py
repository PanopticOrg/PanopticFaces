"""French official sources for the gallery: Assemblée nationale, Sénat, and the
Répertoire national des élus (RNE).

    deputies()   every deputy since the 14th legislature (2012), from the Assemblée's
                 open data: name, birth date, legislatures, party of financial
                 attachment (PARPOL) and parliamentary group (GP), and one official
                 portrait per legislature.
    senators()   every senator listed by the Sénat (data.senat.fr): name, birth date,
                 political group, active or not. Portraits are addressed by the
                 senat.fr id, which Wikidata holds (P1808).
    mayors()     mayors in the RNE of the given communes (INSEE codes): name, birth
                 date, commune. The RNE has neither photos nor parties.

Downloads are cached under <cache>/official.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path
from typing import Callable

AN_ZIP = ('https://data.assemblee-nationale.fr/static/openData/repository/17/amo/'
          'tous_acteurs_mandats_organes_xi_legislature/'
          'AMO30_tous_acteurs_tous_mandats_tous_organes_historique.json.zip')
AN_PHOTO = 'https://www2.assemblee-nationale.fr/static/tribun/{leg}/photos/{num}.jpg'
AN_PAGE = 'https://www.assemblee-nationale.fr/dyn/deputes/PA{num}'
MIN_LEGISLATURE = 14                         # 2012 onwards, as the Wikidata groups

SENAT_CSV = 'https://data.senat.fr/data/senateurs/ODSEN_GENERAL.csv'
SENAT_PHOTO = 'https://www.senat.fr/senimg/{slug}_carre.jpg'
SENAT_PAGE = 'https://www.senat.fr/senateur/{slug}.html'

RNE_DATASET = 'https://www.data.gouv.fr/api/1/datasets/repertoire-national-des-elus-1/'

Fetch = Callable[..., bytes]


def _cached(cache: Path, name: str, url: str, fetch: Fetch) -> bytes:
    path = cache / 'official' / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(fetch(url))
    return path.read_bytes()


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


# ---------------------------------------------------------------------------
# Assemblée nationale
# ---------------------------------------------------------------------------

def deputies(cache: Path, fetch: Fetch) -> dict[str, dict]:
    """Deputies since MIN_LEGISLATURE, by numeric Assemblée id ('1654' for PA1654)."""
    archive = zipfile.ZipFile(io.BytesIO(_cached(cache, 'an_amo30.json.zip', AN_ZIP, fetch)))
    names = archive.namelist()
    organes = {}
    for name in names:
        if '/organe/' in name:
            o = json.loads(archive.read(name))['organe']
            if o.get('codeType') in ('PARPOL', 'GP'):
                organes[o['uid']] = o.get('libelle')

    out: dict[str, dict] = {}
    for name in names:
        if '/acteur/' not in name:
            continue
        a = json.loads(archive.read(name))['acteur']
        mandates = _as_list(a.get('mandats', {}).get('mandat'))
        legislatures = sorted({int(m['legislature']) for m in mandates
                               if m.get('typeOrgane') == 'ASSEMBLEE' and m.get('legislature')})
        legislatures = [leg for leg in legislatures if leg >= MIN_LEGISLATURE]
        if not legislatures:
            continue
        num = a['uid']['#text'][2:]
        ident = a['etatCivil']['ident']
        out[num] = {
            'name': f"{ident['prenom']} {ident['nom']}",
            'birth': (a['etatCivil'].get('infoNaissance') or {}).get('dateNais'),
            'legislatures': legislatures,
            'party': _latest(mandates, 'PARPOL', organes),
            'group': _latest(mandates, 'GP', organes),
            'photos': [{'title': f'Assemblée nationale PA{num} - {leg}e législature',
                        'thumb': AN_PHOTO.format(leg=leg, num=num),
                        'page': AN_PAGE.format(num=num),
                        'license': 'Assemblée nationale (portrait officiel)',
                        'artist': 'Assemblée nationale'}
                       for leg in reversed(legislatures)],
        }
    return out


def _latest(mandates: list[dict], kind: str, organes: dict[str, str]) -> str | None:
    """Label of the most recent mandate of type ``kind`` (an open one first)."""
    found = [m for m in mandates if m.get('typeOrgane') == kind]
    if not found:
        return None
    found.sort(key=lambda m: (m.get('dateFin') is None, m.get('dateDebut') or ''))
    ref = (found[-1].get('organes') or {}).get('organeRef')
    return organes.get(ref)


# ---------------------------------------------------------------------------
# Sénat
# ---------------------------------------------------------------------------

def senators(cache: Path, fetch: Fetch) -> dict[str, dict]:
    """Senators by matricule (lower case, e.g. '21071f')."""
    text = _cached(cache, 'senat_general.csv', SENAT_CSV, fetch).decode('latin-1')
    lines = [line for line in text.splitlines() if not line.startswith('%')]
    out = {}
    for row in csv.DictReader(lines):
        out[row['Matricule'].strip().lower()] = {
            'name': f"{row['Prénom usuel'].strip()} {row['Nom usuel'].strip()}",
            'birth': (row.get('Date naissance') or '')[:10] or None,
            'group': row.get('Groupe politique') or None,
            'active': row.get('État') == 'ACTIF',
        }
    return out


def senator_photo(slug: str) -> dict:
    return {'title': f'Sénat {slug} - portrait officiel',
            'thumb': SENAT_PHOTO.format(slug=slug),
            'page': SENAT_PAGE.format(slug=slug),
            'license': 'Sénat (portrait officiel)', 'artist': 'Sénat'}


# ---------------------------------------------------------------------------
# Répertoire national des élus
# ---------------------------------------------------------------------------

def mayors(cache: Path, fetch: Fetch, insee_codes: set[str]) -> list[dict]:
    """Current mayors of the communes ``insee_codes``, from the RNE."""
    meta = json.loads(fetch(RNE_DATASET))
    url = next(r['url'] for r in meta['resources'] if 'maire' in r['url'])
    text = _cached(cache, 'rne_maires.csv', url, fetch).decode('utf-8-sig')
    out = []
    for row in csv.DictReader(io.StringIO(text), delimiter=';'):
        if row['Code de la commune'] in insee_codes:
            out.append({'insee': row['Code de la commune'], 'commune': row['Libellé de la commune'],
                        'last': row["Nom de l'élu"], 'first': row["Prénom de l'élu"],
                        'birth': row['Date de naissance'] or None})
    return out
