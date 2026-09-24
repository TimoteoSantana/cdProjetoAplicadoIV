import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd


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

# def plot_series(df, title=None, xlabel=None, ylabel="Valor", figsize=(14, 6), grid=True):
#     """
#     Plota todas as colunas numéricas de um DataFrame no mesmo gráfico.

#     Parâmetros
#     ----------
#     df : pd.DataFrame
#         DataFrame cujo índice representa o tempo e cada coluna é uma série temporal.
#     title : str, optional
#         Título do gráfico.
#     xlabel : str, optional
#         Rótulo do eixo x. Se None, usa o nome do índice ou 'Tempo'.
#     ylabel : str, optional
#         Rótulo do eixo y.
#     figsize : tuple, optional
#         Tamanho da figura, por padrão (14, 6).
#     grid : bool, default=True
#         Se True, adiciona uma grade vertical sutil nos ticks do tempo.
#     """
#     dados = df.select_dtypes(include="number").copy()

#     if dados.empty:
#         raise ValueError("O DataFrame não possui colunas numéricas para plotar.")

#     fig, ax = plt.subplots(figsize=figsize)

#     for coluna in dados.columns:
#         sns.lineplot(
#             data=dados,
#             x=dados.index,
#             y=coluna,
#             ax=ax,
#             label=coluna
#         )

#     # Remove as molduras superior e direita
#     sns.despine(ax=ax, top=True, right=True)

#     # Inclina os ticks do eixo x
#     ax.tick_params(axis="x", rotation=40)

#     # Grade exclusivamente vertical, vinculada aos ticks principais do eixo x.
#     if grid:
#         ax.grid(
#             axis="x",
#             which="major",
#             color="#818080FF",
#             linestyle="--",
#             linewidth=0.7,
#             alpha=0.65
#         )

#         # Evita qualquer grade horizontal.
#         ax.grid(axis="y", visible=False)

#         # Coloca a grade atrás das linhas das séries.
#         ax.set_axisbelow(True)

#     ax.set_title(title or "Séries temporais")
#     ax.set_xlabel(xlabel or dados.index.name or "Tempo")
#     ax.set_ylabel(ylabel)

#     ax.legend(title="Série", bbox_to_anchor=(1.02, 1), loc="upper left")
#     fig.tight_layout()

#     return fig, ax