"""Forecasting models and evaluation helpers used by the project notebook."""

import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


DEFAULT_SARIMAX_GRID = (
    ((1, 0, 0), (1, 0, 0, 12)),
    ((1, 1, 0), (1, 0, 0, 12)),
    ((0, 1, 1), (0, 1, 1, 12)),
    ((1, 1, 1), (0, 1, 1, 12)),
    ((1, 0, 1), (0, 1, 1, 12)),
    ((1, 1, 0), (1, 1, 0, 12)),
)


def _validate_series(serie: pd.Series, nome: str = "serie") -> None:
    if not isinstance(serie, pd.Series):
        raise TypeError(f"{nome} deve ser uma pd.Series.")
    if not isinstance(serie.index, pd.DatetimeIndex):
        raise TypeError(f"{nome} deve ter DatetimeIndex.")
    if serie.empty or serie.index.has_duplicates or not serie.index.is_monotonic_increasing:
        raise ValueError(f"{nome} deve ser não vazia, única e cronológica.")
    valores = serie.to_numpy(dtype=float)
    if not np.isfinite(valores).all():
        raise ValueError(f"{nome} contém valores ausentes ou não finitos.")


def naive_seasonal_forecast(
    treino: pd.Series,
    horizonte: int = 12,
    periodo: int = 12,
) -> pd.Series:
    """Repeat the last complete seasonal cycle for the requested horizon."""
    _validate_series(treino, "treino")
    if not isinstance(horizonte, int) or horizonte < 1:
        raise ValueError("horizonte deve ser um inteiro positivo.")
    if not isinstance(periodo, int) or periodo < 1 or len(treino) < periodo:
        raise ValueError("O treino deve conter ao menos um ciclo sazonal completo.")
    inicio = treino.index[-1] + pd.offsets.MonthBegin(1)
    datas = pd.date_range(start=inicio, periods=horizonte, freq="MS")
    valores = np.resize(treino.iloc[-periodo:].to_numpy(dtype=float), horizonte)
    return pd.Series(valores, index=datas, name=treino.name)


def evaluate_forecast(
    real: pd.Series,
    previsto: pd.Series,
    treino: pd.Series,
    periodo: int = 12,
) -> dict[str, float]:
    """Calculate MAE, RMSE, sMAPE, and MASE on aligned monthly observations."""
    for serie, nome in ((real, "real"), (previsto, "previsto"), (treino, "treino")):
        _validate_series(serie, nome)
    if not real.index.equals(previsto.index):
        raise ValueError("As datas de real e previsto devem estar alinhadas.")
    if not isinstance(periodo, int) or periodo < 1 or len(treino) <= periodo:
        raise ValueError("Treino insuficiente para calcular MASE no período informado.")

    valores_reais = real.to_numpy(dtype=float)
    valores_previstos = previsto.to_numpy(dtype=float)
    erro = valores_reais - valores_previstos
    denominador = np.abs(valores_reais) + np.abs(valores_previstos)
    smape = np.divide(
        200 * np.abs(erro),
        denominador,
        out=np.zeros_like(erro),
        where=denominador != 0,
    )
    escala = np.mean(np.abs(
        treino.to_numpy(dtype=float)[periodo:] -
        treino.to_numpy(dtype=float)[:-periodo]
    ))

    return {
        "MAE": float(np.mean(np.abs(erro))),
        "RMSE": float(np.sqrt(np.mean(np.square(erro)))),
        "sMAPE": float(np.mean(smape)),
        "MASE": float(np.mean(np.abs(erro)) / escala) if escala > 0 else np.nan,
    }


def fit_sarimax(
    serie: pd.Series,
    ordem: tuple[int, int, int],
    ordem_sazonal: tuple[int, int, int, int],
):
    """Fit a SARIMAX model and return its fitted result and warning messages."""
    _validate_series(serie)
    with warnings.catch_warnings(record=True) as avisos:
        warnings.simplefilter("always")
        modelo = SARIMAX(
            serie.astype(float),
            order=ordem,
            seasonal_order=ordem_sazonal,
            trend="c" if ordem[1] + ordem_sazonal[1] == 0 else "n",
            enforce_stationarity=True,
            enforce_invertibility=True,
        )
        ajustado = modelo.fit(disp=False, maxiter=500)
    if not ajustado.mle_retvals.get("converged", False):
        raise RuntimeError(f"SARIMAX não convergiu para {serie.name!r}.")
    return ajustado, [str(aviso.message) for aviso in avisos]


def forecast_sarimax(ajustado, horizonte: int = 12) -> pd.Series:
    if not isinstance(horizonte, int) or horizonte < 1:
        raise ValueError("horizonte deve ser um inteiro positivo.")
    previsao = ajustado.get_forecast(steps=horizonte).predicted_mean
    valores = np.asarray(previsao, dtype=float)
    if not np.isfinite(valores).all():
        raise ValueError("SARIMAX produziu previsão não finita.")
    indice = previsao.index
    if not isinstance(indice, pd.DatetimeIndex):
        indice_historico = pd.DatetimeIndex(ajustado.model.data.row_labels)
        indice = pd.date_range(
            start=indice_historico[-1] + pd.offsets.MonthBegin(1),
            periods=horizonte,
            freq="MS",
        )
    return pd.Series(
        np.maximum(valores, 0),
        index=pd.DatetimeIndex(indice),
        name=previsao.name,
    )


def search_sarimax(
    treino: pd.Series,
    validacao: pd.Series,
    grade: tuple = DEFAULT_SARIMAX_GRID,
) -> tuple[dict, pd.DataFrame]:
    """Select SARIMAX orders using validation RMSE only."""
    _validate_series(treino, "treino")
    _validate_series(validacao, "validacao")
    if not grade:
        raise ValueError("A grade SARIMAX não pode ser vazia.")
    if validacao.index[0] <= treino.index[-1]:
        raise ValueError("A validação deve ocorrer depois do treino.")

    auditoria = []
    candidatos = []
    for ordem, ordem_sazonal in grade:
        registro = {
            "order": ordem,
            "seasonal_order": ordem_sazonal,
            "status": "falhou",
        }
        try:
            ajustado, avisos = fit_sarimax(treino, ordem, ordem_sazonal)
            previsto = forecast_sarimax(ajustado, len(validacao))
            previsto.index = validacao.index
            metricas = evaluate_forecast(validacao, previsto, treino)
            registro.update(status="ok", RMSE_validacao=metricas["RMSE"], avisos=avisos)
            candidatos.append((metricas["RMSE"], ordem, ordem_sazonal, avisos))
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            registro["erro"] = str(exc)
        auditoria.append(registro)

    if not candidatos:
        raise RuntimeError(
            "Nenhum candidato SARIMAX convergiu na validação. "
            f"Auditoria: {auditoria}"
        )
    candidatos.sort(key=lambda item: item[0])
    melhor_rmse, melhor_ordem, melhor_sazonal, avisos = candidatos[0]
    selecionado = {
        "order": melhor_ordem,
        "seasonal_order": melhor_sazonal,
        "RMSE_validacao": melhor_rmse,
        "avisos": avisos,
    }
    return selecionado, pd.DataFrame(auditoria)


def _prophet_class():
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError(
            "Prophet é necessário para executar esta etapa. "
            "Instale as dependências declaradas em projeto/requirements.txt."
        ) from exc
    return Prophet


def fit_prophet(
    serie: pd.Series,
    ordem_fourier: int = 5,
    changepoint_prior_scale: float = 0.05,
):
    """Fit additive Prophet with monthly-appropriate yearly seasonality."""
    _validate_series(serie)
    if not isinstance(ordem_fourier, int) or ordem_fourier < 1:
        raise ValueError("ordem_fourier deve ser um inteiro positivo.")
    if not np.isfinite(changepoint_prior_scale) or changepoint_prior_scale <= 0:
        raise ValueError("changepoint_prior_scale deve ser positivo e finito.")

    Prophet = _prophet_class()
    dados = pd.DataFrame({"ds": serie.index, "y": serie.to_numpy(dtype=float)})
    modelo = Prophet(
        yearly_seasonality=ordem_fourier,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=changepoint_prior_scale,
        uncertainty_samples=0,
    )
    modelo.fit(dados, seed=42)
    return modelo


def forecast_prophet(modelo, horizonte: int = 12) -> pd.Series:
    if not isinstance(horizonte, int) or horizonte < 1:
        raise ValueError("horizonte deve ser um inteiro positivo.")
    futuro = modelo.make_future_dataframe(
        periods=horizonte,
        freq="MS",
        include_history=False,
    )
    previsao = modelo.predict(futuro)
    valores = previsao["yhat"].to_numpy(dtype=float)
    if not np.isfinite(valores).all():
        raise ValueError("Prophet produziu previsão não finita.")
    datas = pd.DatetimeIndex(previsao["ds"])
    return pd.Series(np.maximum(valores, 0), index=datas, name="yhat")


def search_prophet(
    treino: pd.Series,
    validacao: pd.Series,
    ordens_fourier: tuple[int, ...] = (3, 5, 10),
    changepoint_scales: tuple[float, ...] = (0.01, 0.05, 0.1),
) -> tuple[dict, pd.DataFrame]:
    """Select Prophet yearly smoothness and trend flexibility by validation RMSE."""
    _validate_series(treino, "treino")
    _validate_series(validacao, "validacao")
    if validacao.index[0] <= treino.index[-1]:
        raise ValueError("A validação deve ocorrer depois do treino.")
    if not ordens_fourier or not changepoint_scales:
        raise ValueError("As grades Prophet não podem ser vazias.")

    auditoria = []
    candidatos = []
    for ordem_fourier, changepoint_scale in itertools.product(
        ordens_fourier, changepoint_scales
    ):
        modelo = fit_prophet(treino, ordem_fourier, changepoint_scale)
        previsto = forecast_prophet(modelo, len(validacao))
        previsto.index = validacao.index
        metricas = evaluate_forecast(validacao, previsto, treino)
        registro = {
            "yearly_fourier_order": ordem_fourier,
            "changepoint_prior_scale": changepoint_scale,
            "RMSE_validacao": metricas["RMSE"],
        }
        auditoria.append(registro)
        candidatos.append((metricas["RMSE"], ordem_fourier, changepoint_scale))

    melhor_rmse, melhor_fourier, melhor_changepoint = min(candidatos)
    selecionado = {
        "yearly_fourier_order": melhor_fourier,
        "changepoint_prior_scale": melhor_changepoint,
        "RMSE_validacao": melhor_rmse,
    }
    return selecionado, pd.DataFrame(auditoria)
