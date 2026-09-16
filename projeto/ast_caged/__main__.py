from __future__ import annotations

import argparse
import sys
import traceback

from ast_caged.modules.data_source import caged_extraction
from ast_caged.modules.data_source import caged_catalog
from ast_caged.modules.utils.utils import exception_trackers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ast_caged",
        description="CLI para extração e processamento de microdados do Novo CAGED.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        title="comandos",
        description="Comandos disponíveis.",
    )

    parser_extraction = subparsers.add_parser(
        "extraction",
        help="Executa a extração e o processamento de microdados do CAGED.",
    )

    parser_extraction.set_defaults(func=caged_extraction.main)

    parser_catalog = subparsers.add_parser(
        "catalog",
        help="Baixa o catálogo do CAGED e salva cada aba como CSV.",
    )
    parser_catalog.set_defaults(func=caged_catalog.main)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.func()
    except KeyboardInterrupt:
        print("\nOperação cancelada pelo usuário.")
        return 1
    except Exception as exc:
        print(f"Erro durante a execução: {exc}", file=sys.stderr)
        exception_trackers(exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())