import unittest

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from ast_caged.modules.inference.datasplit import temporal_split
from ast_caged.modules.inference.prediction import (
    evaluate_forecast,
    fit_sarimax,
    naive_seasonal_forecast,
    forecast_sarimax,
)
from ast_caged.modules.visualization.plotters import plot_forecast


def sample_series(length: int = 79) -> pd.Series:
    index = pd.date_range("2020-01-01", periods=length, freq="MS")
    values = np.arange(length, dtype=float)
    return pd.Series(values, index=index, name="admissoes")


class TemporalSplitTests(unittest.TestCase):
    def test_split_keeps_annual_validation_and_test_horizons(self):
        data = pd.DataFrame({"admissoes": sample_series().to_numpy()},
                            index=sample_series().index)

        split = temporal_split("Minas Gerais", data)

        self.assertEqual(len(split.treino), 55)
        self.assertEqual(len(split.validacao), 12)
        self.assertEqual(len(split.teste), 12)
        self.assertEqual(split.periodos.loc[0, "uf"], "Minas Gerais")
        self.assertEqual(split.periodos.loc[0, "meses_treino"], 55)
        self.assertLess(split.treino.index.max(), split.validacao.index.min())
        self.assertLess(split.validacao.index.max(), split.teste.index.min())

    def test_split_rejects_missing_month(self):
        series = sample_series()
        series = series.drop(series.index[20])
        data = pd.DataFrame({"admissoes": series.to_numpy()}, index=series.index)

        with self.assertRaisesRegex(ValueError, "sem lacunas"):
            temporal_split("ES", data)

    def test_split_rejects_insufficient_history(self):
        series = sample_series(59)
        data = pd.DataFrame({"admissoes": series.to_numpy()},
                            index=series.index)

        with self.assertRaisesRegex(ValueError, "são necessários"):
            temporal_split("RJ", data)


class ForecastingTests(unittest.TestCase):
    def test_naive_forecast_repeats_last_year_and_extends_months(self):
        series = sample_series(24)

        forecast = naive_seasonal_forecast(series, horizonte=14)

        np.testing.assert_array_equal(forecast.iloc[:12], series.iloc[-12:])
        np.testing.assert_array_equal(forecast.iloc[12:],
                                      series.iloc[-12:-10].to_numpy())
        self.assertEqual(forecast.index[0], pd.Timestamp("2022-01-01"))
        self.assertEqual(forecast.index[-1], pd.Timestamp("2023-02-01"))

    def test_metrics_handle_zero_smape_denominator(self):
        dates = pd.date_range("2020-01-01", periods=14, freq="MS")
        treino = pd.Series(np.arange(14, dtype=float), index=dates)
        real = pd.Series([0.0, 2.0], index=dates[-2:])
        previsto = pd.Series([0.0, 1.0], index=dates[-2:])

        metrics = evaluate_forecast(real, previsto, treino)

        self.assertEqual(metrics["MAE"], 0.5)
        self.assertAlmostEqual(metrics["RMSE"], np.sqrt(0.5))
        self.assertEqual(metrics["sMAPE"], 100.0 / 3.0)

    def test_sarimax_forecast_uses_monthly_dates_without_index_frequency(self):
        series = sample_series(60)
        fitted, _ = fit_sarimax(series, (1, 0, 0), (0, 0, 0, 0))

        forecast = forecast_sarimax(fitted, 3)

        self.assertEqual(
            list(forecast.index),
            list(pd.date_range("2025-01-01", periods=3, freq="MS")),
        )

    def test_plot_marks_observation_and_forecast(self):
        history = sample_series(24)
        forecast = naive_seasonal_forecast(history, 3)

        figure, axis = plot_forecast(history, forecast)

        self.assertEqual(len(axis.lines), 3)
        self.assertEqual(axis.lines[0].get_label(), "Observado")
        self.assertEqual(axis.lines[1].get_label(), "Previsto")
        matplotlib.pyplot.close(figure)


if __name__ == "__main__":
    unittest.main()
