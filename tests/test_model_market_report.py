import unittest
import pandas as pd
from model_market_report import build_model_market_report


class ModelMarketReportTests(unittest.TestCase):
    def setUp(self):
        self.games = pd.DataFrame([
            {"season": 2022, "market_available": True, "home_score": 24, "away_score": 17,
             "model_favorite_win_probability": 0.65, "favorite_no_vig_market_probability": 0.60,
             "actual_favorite_won": True, "bet_taken": True, "bet_result": 1, "risked_units": 1.0, "net_units": 0.8},
            {"season": 2022, "market_available": True, "home_score": 10, "away_score": 20,
             "model_favorite_win_probability": 0.55, "favorite_no_vig_market_probability": 0.58,
             "actual_favorite_won": False, "bet_taken": False, "bet_result": None, "risked_units": 0.0, "net_units": 0.0},
            {"season": 2023, "market_available": True, "home_score": 13, "away_score": 13,
             "model_favorite_win_probability": 0.62, "favorite_no_vig_market_probability": 0.57,
             "actual_favorite_won": None, "bet_taken": True, "bet_result": None, "risked_units": 1.0, "net_units": 0.0},
            {"season": 2023, "market_available": False, "home_score": 21, "away_score": 14,
             "model_favorite_win_probability": 0.70, "favorite_no_vig_market_probability": None,
             "actual_favorite_won": True, "bet_taken": False, "bet_result": None, "risked_units": 0.0, "net_units": 0.0},
        ])

    def test_scores_same_games_and_excludes_ties_and_missing_odds(self):
        report = build_model_market_report(self.games)
        self.assertEqual(int(report["summary"].iloc[0]["Games scored"]), 2)
        self.assertEqual(set(report["by_season"]["Season"]), {2022})
        self.assertEqual(int(report["bets"].iloc[0]["Bets"]), 2)
        self.assertEqual(int(report["bets"].iloc[0]["Pushes"]), 1)

    def test_empty_input_returns_empty_tables(self):
        report = build_model_market_report(pd.DataFrame())
        self.assertTrue(report["summary"].empty)
        self.assertTrue(report["by_season"].empty)
        self.assertTrue(report["by_edge"].empty)

    def test_market_probability_comparison_uses_same_outcomes(self):
        report = build_model_market_report(self.games)
        row = report["summary"].iloc[0]
        self.assertAlmostEqual(row["Elo Brier (lower better)"], ((0.65 - 1) ** 2 + (0.55 - 0) ** 2) / 2)
        self.assertAlmostEqual(row["Market Brier (lower better)"], ((0.60 - 1) ** 2 + (0.58 - 0) ** 2) / 2)


if __name__ == "__main__":
    unittest.main()
