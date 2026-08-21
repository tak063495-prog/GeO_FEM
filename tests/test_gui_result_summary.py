from __future__ import annotations

import unittest

from geofem_app.gui.result_summary import build_result_judgment_summary


class ResultJudgmentSummaryTests(unittest.TestCase):
    def test_confirmed_srm_reports_bracket_precision_trials_and_elapsed(self) -> None:
        model = build_result_judgment_summary(
            {
                "analysis": "plane_strain_static",
                "stages": [
                    {
                        "name": "Case2 SRM",
                        "max_displacement": 0.0626982513,
                        "solver": {
                            "converged": True,
                            "performance": {"elapsed_seconds": 3600.0},
                            "srm": {
                                "factor_of_safety": 1.284375,
                                "stable_factor": 1.284375,
                                "failed_factor": 1.2875,
                                "factor_tol": 0.005,
                                "search_mode": "auto",
                                "trial_timing": {"total_elapsed_seconds": 3723.0},
                                "trials": [{"factor": value} for value in range(9)],
                            },
                        },
                    }
                ],
            },
            locale="ja",
        )

        metrics = dict(model["metrics"])
        self.assertEqual(model["kind"], "srm")
        self.assertEqual(model["tone"], "success")
        self.assertTrue(model["confirmed"])
        self.assertIn("FOS=1.284375", model["headline"])
        self.assertEqual(metrics["判定区間"], "1.284375 - 1.2875")
        self.assertIn("許容 0.005 / 達成", metrics["探索精度"])
        self.assertEqual(metrics["試行数"], "9")
        self.assertEqual(metrics["解析時間"], "1時間02分03秒")

    def test_confirmed_srm_below_one_is_a_visible_danger(self) -> None:
        model = build_result_judgment_summary(
            {
                "stages": [
                    {
                        "name": "Case1 SRM",
                        "solver": {
                            "converged": True,
                            "srm": {
                                "factor_of_safety": 0.69153,
                                "stable_factor": 0.69153,
                                "failed_factor": 0.69528,
                                "factor_tol": 0.005,
                                "trials": [],
                            },
                        },
                    }
                ]
            },
            locale="ja",
        )

        self.assertEqual(model["tone"], "danger")
        self.assertIn("1.0未満", model["warning"])

    def test_unbounded_srm_is_not_presented_as_a_final_fos(self) -> None:
        model = build_result_judgment_summary(
            {
                "stages": [
                    {
                        "name": "SRM",
                        "solver": {
                            "converged": True,
                            "srm": {
                                "factor_of_safety": 2.5,
                                "stable_factor": 2.5,
                                "failed_factor": None,
                                "trials": [{"factor": 2.5}],
                            },
                        },
                    }
                ]
            },
            locale="ja",
        )

        self.assertEqual(model["tone"], "warning")
        self.assertFalse(model["confirmed"])
        self.assertIn("FOS>=2.5", model["headline"])
        self.assertIn("確定値ではありません", model["warning"])

    def test_standard_analysis_summary_prioritizes_convergence(self) -> None:
        model = build_result_judgment_summary(
            {
                "analysis": "plane_strain_static",
                "warnings": [],
                "stages": [
                    {
                        "name": "Stage-1",
                        "max_displacement": 0.0125,
                        "max_settlement": 0.004,
                        "solver": {
                            "converged": True,
                            "residual_norm": 1.0e-8,
                            "iterations": 4,
                            "performance": {"elapsed_seconds": 65.0},
                        },
                    }
                ],
            },
            locale="ja",
        )

        metrics = dict(model["metrics"])
        self.assertEqual(model["kind"], "analysis")
        self.assertEqual(model["tone"], "success")
        self.assertEqual(model["headline"], "解析は収束しました")
        self.assertEqual(metrics["解析時間"], "1分05秒")
        self.assertEqual(metrics["反復回数"], "4")


if __name__ == "__main__":
    unittest.main()
