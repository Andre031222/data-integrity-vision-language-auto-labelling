"""
Bulk-download alpaca imagery from iNaturalist.

Calls the iNaturalist v1 API directly, with no wrapper, and fetches
CC-licensed Lama pacos observations either worldwide or restricted to Peru.

Verified taxon ids:
  319688  = Lama pacos (alpaca)  -- 4,729 worldwide / 815 in Peru
  42237   = Lama (genus)         -- 20,594 worldwide / 2,654 in Peru
  1624576 = Lama vicugna         -- 3,407 worldwide / 650 in Peru

Usage:
    python src/data/download_inaturalist.py --max_images 800 --country PE
    python src/data/download_inaturalist.py --max_images 4000 --country ALL
"""

import argparse
import json
import time
from pathlib import Path

import requests
from tqdm import tqdm

INAT_API = "https://api.inaturalist.org/v1/observations"

# Taxon IDs verificados
TAXON_IDS = {
    "alpaca":  319688,   # Lama pacos -- solo alpacas
    "lama":    42237,    # Lama genus -- alpaca + llama (más imágenes)
    "vicugna": 1624576,  # Lama vicugna -- vicuña
}

# place_id verificados en iNaturalist
PLACE_IDS = {
    "PE":  7513,   # Perú
    "BO":  7174,   # Bolivia
    "AR":  6966,   # Argentina
    "ALL": None,   # Sin filtro -- máximo volumen
}

DEFAULT_LICENSES = "cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa,cc0"


def _fetch_page(taxon_id: int, place_id, quality_grade: str, page: int, per_page: int) -> dict:
    params = {
        "taxon_id":     taxon_id,
        "quality_grade": quality_grade,
        "per_page":     per_page,
        "page":         page,
        "order":        "desc",
        "order_by":     "votes",
        "license":      DEFAULT_LICENSES,
    }
    if place_id:
        params["place_id"] = place_id

    r = requests.get(INAT_API, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def download_inaturalist_alpacas(
    max_images: int = 800,
    output_dir: str = "data/raw/inaturalist",
    quality_grade: str = "any",   # 'research' | 'needs_id' | 'any'
    country: str = "PE",
    taxon: str = "alpaca",        # 'alpaca' | 'lama' | 'vicugna'
) -> int:
    taxon_id = TAXON_IDS[taxon]
    place_id = PLACE_IDS.get(country, PLACE_IDS["PE"])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Contar disponibles antes de empezar
    try:
        check = _fetch_page(taxon_id, place_id, quality_grade, page=1, per_page=1)
        total_available = check.get("total_results", 0)
        to_download = min(max_images, total_available)
        print(f"iNaturalist | taxon={taxon} (id={taxon_id}) | pais={country} | calidad={quality_grade}")
        print(f"Disponibles: {total_available:,}  |  A descargar: {to_download:,}")
    except Exception as e:
        print(f"Error al consultar API: {e}")
        return 0

    metadata = []
    downloaded = 0
    page = 1
    per_page = 100  # seguro para no saturar

    with tqdm(total=to_download, desc="Descargando", unit="img") as pbar:
        while downloaded < to_download:
            try:
                data = _fetch_page(taxon_id, place_id, quality_grade, page, per_page)
            except Exception as e:
                print(f"\nError API página {page}: {e}")
                time.sleep(5)
                page += 1
                continue

            results = data.get("results", [])
            if not results:
                break

            for obs in results:
                if downloaded >= to_download:
                    break

                photos = obs.get("photos", [])
                if not photos:
                    continue

                photo = photos[0]
                raw_url = photo.get("url", "")
                if not raw_url:
                    continue

                # iNaturalist sirve fotos en varios tamaños: square/thumb/small/medium/large/original
                photo_url = raw_url.replace("square", "medium")

                obs_id  = obs.get("id")
                photo_id = photo.get("id")
                filename = f"inat_{obs_id}_{photo_id}.jpg"
                filepath = output_path / filename

                if filepath.exists():
                    downloaded += 1
                    pbar.update(1)
                    continue

                try:
                    resp = requests.get(photo_url, timeout=20)
                    if resp.status_code == 200 and len(resp.content) > 5000:
                        filepath.write_bytes(resp.content)
                        loc = obs.get("location") or ""
                        parts = loc.split(",") if loc else []
                        metadata.append({
                            "filename":      filename,
                            "obs_id":        obs_id,
                            "photo_id":      photo_id,
                            "license":       photo.get("license_code"),
                            "url":           photo_url,
                            "lat":           parts[0] if len(parts) > 0 else None,
                            "lon":           parts[1] if len(parts) > 1 else None,
                            "quality_grade": obs.get("quality_grade"),
                            "date":          obs.get("observed_on"),
                            "place":         obs.get("place_guess"),
                            "taxon":         taxon,
                            "source":        "inaturalist",
                        })
                        downloaded += 1
                        pbar.update(1)
                        time.sleep(0.15)  # respetar rate limit (~100 req/min)
                    else:
                        pass  # imagen muy pequeña o error HTTP
                except Exception:
                    pass

            page += 1
            time.sleep(1)

    metadata_path = output_path / "metadata.json"
    existing = []
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            existing = json.load(f)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(existing + metadata, f, ensure_ascii=False, indent=2)

    print(f"\nDescarga completa: {downloaded} imagenes en '{output_dir}'")
    print(f"Metadata: {metadata_path}")
    return downloaded


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Descargar alpacas desde iNaturalist")
    parser.add_argument("--max_images", type=int, default=800)
    parser.add_argument("--output",     type=str, default="data/raw/inaturalist")
    parser.add_argument("--quality",    type=str, default="any",
                        choices=["research", "needs_id", "any"])
    parser.add_argument("--country",    type=str, default="PE",
                        choices=["PE", "BO", "AR", "ALL"])
    parser.add_argument("--taxon",      type=str, default="alpaca",
                        choices=["alpaca", "lama", "vicugna"])
    args = parser.parse_args()

    download_inaturalist_alpacas(
        max_images=args.max_images,
        output_dir=args.output,
        quality_grade=args.quality,
        country=args.country,
        taxon=args.taxon,
    )
