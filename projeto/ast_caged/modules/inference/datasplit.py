"""Chronological train, validation, and test splits for monthly series."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    treino: pd.DataFrame
    validacao: pd.DataFrame
    teste: pd.DataFrame
    periodos: pd.DataFrame


def temporal_split(
    nome_serie: str,
    dados: pd.DataFrame,
    horizonte_teste: int = 12,
    horizonte_validacao: int = 12,
    minimo_treino: int = 36,
) -> TemporalSplit:
    """Split monthly data chronologically and describe the resulting periods."""
    if not isinstance(dados, pd.DataFrame):
        raise TypeError("dados deve ser um DataFrame.")
    if not isinstance(dados.index, pd.DatetimeIndex):
        raise TypeError("O índice de dados deve ser um DatetimeIndex.")
    if dados.empty:
        raise ValueError(f"A série {nome_serie!r} está vazia.")
    if dados.index.has_duplicates or not dados.index.is_monotonic_increasing:
        raise ValueError("O índice deve ser único e estar em ordem cronológica.")
    if not all(isinstance(valor, int) and valor > 0 for valor in (
        horizonte_teste, horizonte_validacao, minimo_treino
    )):
        raise ValueError("Horizontes e mínimo de treino devem ser inteiros positivos.")

    indice_esperado = pd.date_range(
        start=dados.index[0], periods=len(dados), freq="MS"
    )
    if not dados.index.equals(indice_esperado):
        raise ValueError("A série deve ter uma observação por mês, sem lacunas.")

    n_treino = len(dados) - horizonte_validacao - horizonte_teste
    if n_treino < minimo_treino:
        minimo = minimo_treino + horizonte_validacao + horizonte_teste
        raise ValueError(
            f"A série {nome_serie!r} tem {len(dados)} meses; são necessários "
            f"ao menos {minimo} ({minimo_treino} treino + "
            f"{horizonte_validacao} validação + {horizonte_teste} teste)."
        )

    treino = dados.iloc[:n_treino].copy()
    validacao = dados.iloc[n_treino:n_treino + horizonte_validacao].copy()
    teste = dados.iloc[n_treino + horizonte_validacao:].copy()
    periodos = pd.DataFrame([{
        "uf": nome_serie,
        "treino_inicio": treino.index.min(),
        "treino_fim": treino.index.max(),
        "teste_inicio": teste.index.min(),
        "teste_fim": teste.index.max(),
        "meses_treino": len(treino),
        "validacao_inicio": validacao.index.min(),
        "validacao_fim": validacao.index.max(),
    }])

    return TemporalSplit(
        treino=treino,
        validacao=validacao,
        teste=teste,
        periodos=periodos,
    )
