from __future__ import annotations

import json
import re
import tempfile
from ftplib import FTP
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PATHS_CONFIG_FILE = PACKAGE_ROOT / "config" / "paths.json"

FTP_HOST = "ftp.mtps.gov.br"
FTP_DIRECTORY = "/pdet/microdados/NOVO CAGED"
CATALOG_FILE_NAME = "Layout Não-identificado Novo Caged Movimentação.xlsx"


def load_json(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {file_path}")

    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_project_path(relative_path: str) -> Path:
    return (PROJECT_ROOT / relative_path).resolve()


def get_catalog_directory() -> Path:
    paths_config = load_json(PATHS_CONFIG_FILE)
    try:
        relative_path = paths_config["data"]["catalog_caged"]
    except KeyError as exc:
        raise KeyError(
            "A chave 'data.catalog_caged' não foi encontrada em paths.json."
        ) from exc

    return resolve_project_path(relative_path)


def download_catalog(local_file_path: Path) -> Path:
    temporary_file_path = local_file_path.with_suffix(
        f"{local_file_path.suffix}.part"
    )
    temporary_file_path.unlink(missing_ok=True)

    try:
        with FTP(FTP_HOST, timeout=300) as ftp:
            ftp.encoding = "latin-1"
            ftp.login()
            ftp.cwd(FTP_DIRECTORY)
            with temporary_file_path.open("wb") as local_file:
                ftp.retrbinary(
                    f"RETR {CATALOG_FILE_NAME}",
                    local_file.write,
                    blocksize=1024 * 1024,
                )

        temporary_file_path.replace(local_file_path)
    except Exception:
        temporary_file_path.unlink(missing_ok=True)
        raise

    return local_file_path


def sanitize_sheet_name(sheet_name: str) -> str:
    safe_name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", sheet_name).strip()
    safe_name = safe_name.rstrip(".")
    return safe_name or "aba_sem_nome"


def export_catalog(workbook_path: Path, output_directory: Path) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    output_paths = []

    with pd.ExcelFile(workbook_path, engine="openpyxl") as workbook:
        used_names: set[str] = set()

        for sheet_name in workbook.sheet_names:
            output_name = sanitize_sheet_name(sheet_name)
            candidate = output_name
            suffix = 2
            while candidate.casefold() in used_names:
                candidate = f"{output_name}_{suffix}"
                suffix += 1
            used_names.add(candidate.casefold())

            dataframe = pd.read_excel(workbook, sheet_name=sheet_name)
            output_path = output_directory / f"{candidate}.csv"
            dataframe.to_csv(output_path, sep=";", encoding="utf-8", index=False)
            output_paths.append(output_path)
            print(f"[catalog] Aba '{sheet_name}' salva em: {output_path}")

    return output_paths


def main() -> None:
    output_directory = get_catalog_directory()

    with tempfile.TemporaryDirectory(prefix="ast_caged_catalog_") as temporary_directory:
        workbook_path = Path(temporary_directory) / CATALOG_FILE_NAME
        print("[download] Baixando catálogo para processamento...")
        download_catalog(workbook_path)
        output_paths = export_catalog(workbook_path, output_directory)

    print(f"[done] {len(output_paths)} abas exportadas para: {output_directory}")