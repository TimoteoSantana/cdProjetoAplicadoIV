import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def plot_series(
        df, 
        title=None, 
        xlabel=None, 
        ylabel="Valor", 
        figsize=(14, 6), 
        mostrar_grid_anual=True
):
    
    """
    Plota cada coluna numérica do DataFrame como uma linha no mesmo gráfico.

    Espera-se que o índice seja um DatetimeIndex, com uma observação
    por mês de competência.
    """
    dados = df.select_dtypes(include="number").copy()

    if dados.empty:
        raise ValueError("O DataFrame não possui colunas numéricas para plotar.")

    if not isinstance(dados.index, type(__import__("pandas").DatetimeIndex([]))):
        raise TypeError(
            "O índice do DataFrame deve ser um DatetimeIndex."
        )

    fig, ax = plt.subplots(figsize=figsize)

    for coluna in dados.columns:
        sns.lineplot(
            data=dados,
            x=dados.index,
            y=coluna,
            ax=ax,
            label=coluna
        )

    # Remove bordas superior e direita.
    sns.despine(ax=ax, top=True, right=True)

    # Ticks principais: anos, mostrados como rótulo no eixo x.
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Ticks secundários: meses, usados apenas para desenhar o grid.
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    # Rotaciona somente os rótulos anuais.
    ax.tick_params(axis="x", which="major", rotation=40)

    # Esconde os pequenos ticks mensais inferiores e seus possíveis rótulos.
    ax.tick_params(
        axis="x",
        which="minor",
        bottom=False,
        labelbottom=False
    )

    # Uma linha vertical pontilhada para cada mês.
    ax.grid(
        axis="x",
        which="minor",
        color="#7e7e7e",
        linestyle=":",
        linewidth=0.5,
        alpha=0.9
    )

    # Linha anual opcional, levemente mais destacada.
    if mostrar_grid_anual:
        ax.grid(
            axis="x",
            which="major",
            color="#202020",
            linestyle="--",
            linewidth=0.8,
            alpha=0.9
        )

    # Grade sempre atrás das séries.
    ax.set_axisbelow(True)

    ax.set_title(title or "Séries temporais")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(
        title="Série",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    fig.tight_layout()

    return fig, ax


def plot_forecast(
    historico: pd.Series,
    previsao: pd.Series,
    title: str | None = None,
    xlabel: str = "Competência",
    ylabel: str = "Empregos",
    figsize: tuple[float, float] = (14, 6),
):
    """Plot observed history and future forecast with a clear time boundary."""
    for serie, nome in ((historico, "historico"), (previsao, "previsao")):
        if not isinstance(serie, pd.Series):
            raise TypeError(f"{nome} deve ser uma pd.Series.")
        if not isinstance(serie.index, pd.DatetimeIndex):
            raise TypeError(f"{nome} deve ter DatetimeIndex.")
        if serie.empty or serie.index.has_duplicates:
            raise ValueError(f"{nome} deve ser não vazia e sem datas duplicadas.")
        if not serie.index.is_monotonic_increasing:
            raise ValueError(f"{nome} deve estar em ordem cronológica.")
        if not pd.api.types.is_numeric_dtype(serie):
            raise TypeError(f"{nome} deve conter valores numéricos.")
        if not np.isfinite(serie.to_numpy(dtype=float)).all():
            raise ValueError(f"{nome} contém valores ausentes ou não finitos.")
    if historico.index.max() >= previsao.index.min():
        raise ValueError("A previsão deve começar depois do histórico observado.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(historico.index, historico, color="#1f77b4", label="Observado")
    ax.plot(
        previsao.index,
        previsao,
        color="#2ca02c",
        marker="o",
        label="Previsto",
    )
    ax.axvline(
        previsao.index.min(),
        color="#555555",
        linestyle=":",
        linewidth=1,
        label="Início da previsão",
    )
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.tick_params(axis="x", which="major", rotation=40)
    ax.tick_params(axis="x", which="minor", bottom=False, labelbottom=False)
    ax.grid(axis="x", which="minor", color="#7e7e7e", linestyle=":", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_title(title or "Histórico e previsão")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig, ax



def prot_components(componentes: pd.DataFrame, titulo: str | None = None):
    colunas = componentes.select_dtypes(include="number").columns

    if len(colunas) == 0:
        raise ValueError("O DataFrame não contém colunas numéricas.")

    fig, eixos = plt.subplots(
        nrows=len(colunas),
        ncols=1,
        figsize=(12, 2.8 * len(colunas)),
        sharex=True,
        squeeze=False,
    )

    for eixo, coluna in zip(eixos[:, 0], colunas):
        eixo.plot(componentes.index, componentes[coluna], linewidth=1.4)
        eixo.set_title(str(coluna).capitalize())
        eixo.grid(alpha=0.3)

        eixo.spines["top"].set_visible(False)
        eixo.spines["right"].set_visible(False)

    if titulo:
        fig.suptitle(titulo)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()

    plt.show()
    return fig, eixos



def plotar_acf_pacf(
    serie: pd.Series, lags: int = 24, titulo: str = "", diferenciar: bool = False
    ):
    """Plota as funções de Autocorrelação (ACF) e Autocorrelação Parcial (PACF)
    para a série em nível ou diferenciada.
    """
    dados_analise = serie.diff().dropna() if diferenciar else serie.dropna()

    sufixo = " (1ª Diferença)" if diferenciar else ""
    titulo_completo = f"{titulo}{sufixo}" if titulo else f"Série{sufixo}"

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))

    # 1. Gráfico de Autocorrelação (ACF)
    plot_acf(dados_analise, lags=lags, ax=axes[0], alpha=0.05)
    axes[0].set_title(f"ACF: {titulo_completo}", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Defasagens (Lags)", fontsize=10)
    axes[0].set_ylabel("Autocorrelação", fontsize=10)
    axes[0].grid(alpha=0.3)

    # 2. Gráfico de Autocorrelação Parcial (PACF)
    # Usamos method='ywm' (Yule-Walker modificado) para estabilidade
    plot_pacf(dados_analise, lags=lags, ax=axes[1], alpha=0.05, method="ywm")
    axes[1].set_title(f"PACF: {titulo_completo}", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Defasagens (Lags)", fontsize=10)
    axes[1].set_ylabel("Autocorrelação Parcial", fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()