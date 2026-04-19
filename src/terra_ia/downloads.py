from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


def download_file(url: str, dest: Path, label: str = "", *, timeout: int = 180, chunk_size: int = 8192) -> bool:
    if dest.exists():
        size_mb = dest.stat().st_size / 1e6
        print(f"  OK deja present : {dest.name} ({size_mb:.1f} Mo)")
        return True

    print(f"  Telechargement {label} -> {dest.name} ...", end=" ", flush=True)
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        with open(dest, "wb") as file_obj:
            for chunk in response.iter_content(chunk_size):
                file_obj.write(chunk)
        print(f"OK ({dest.stat().st_size / 1e6:.1f} Mo)")
        return True
    except Exception as exc:
        print(f"ERREUR - {exc}")
        if dest.exists():
            dest.unlink()
        return False


def download_all_dalles(urls: list[str], prefix: str, data_dir: Path) -> list[Path]:
    print(f"  Telechargement {len(urls)} dalles {prefix}...")
    paths: list[Path] = []
    ok = 0
    skip = 0
    err = 0

    for index, url in enumerate(urls):
        if "FILENAME=" in url:
            filename = url.split("FILENAME=")[-1].split("&")[0]
        else:
            filename = f"{prefix}_dalle_{index:02d}.tif"

        dest = data_dir / filename
        if dest.exists():
            skip += 1
            paths.append(dest)
            continue

        if download_file(url, dest, f"{prefix} {index + 1}/{len(urls)}"):
            ok += 1
            paths.append(dest)
        else:
            err += 1

    print(f"  {prefix} : {ok} telecharges / {skip} existants / {err} erreurs")
    return paths


def download_parcelles(code: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  OK parcelles presentes : {dest.name}")
        return True

    url = (
        "https://cadastre.data.gouv.fr/bundler/cadastre-etalab"
        f"/communes/{code}/geojson/parcelles"
    )
    print(f"  Telechargement parcelles {code} ...", end=" ", flush=True)
    try:
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        dest.write_bytes(response.content)
        print("OK")
        return True
    except Exception as exc:
        print(f"ERREUR - {exc}")
        return False


def download_dvf_latest_for_commune(
    dest: Path,
    *,
    data_dir: Path,
    commune_code: str,
    years: tuple[str, ...] = ("2023", "2022", "2021", "2020"),
) -> bool:
    if dest.exists():
        print(f"  OK DVF present : {dest.name}")
        return True

    for year in years:
        url = f"https://files.data.gouv.fr/geo-dvf/latest/csv/{year}/departements/73.csv.gz"
        dest_gz = data_dir / f"dvf_73_{year}.csv.gz"
        print(f"  Telechargement DVF Savoie {year} ...", end=" ", flush=True)
        try:
            response = requests.get(url, stream=True, timeout=180)
            response.raise_for_status()
            with open(dest_gz, "wb") as file_obj:
                for chunk in response.iter_content(65536):
                    file_obj.write(chunk)

            df_raw = pd.read_csv(dest_gz, low_memory=False, compression="gzip")
            print(f"OK ({len(df_raw)} transactions dept 73)")

            mask = df_raw.get("code_commune", pd.Series(dtype=str)) == commune_code
            df_commune = df_raw[mask] if mask.any() else df_raw
            df_commune.to_csv(dest, index=False)
            dest_gz.unlink()
            print(f"    {len(df_commune)} transactions pour commune {commune_code}")
            return True
        except Exception as exc:
            print(f"ERREUR - {exc}")
            if dest_gz.exists():
                dest_gz.unlink()

    print("  WARNING DVF indisponible - labels H3/H4 desactives")
    return False
