"""Source-alias contract independent of the training stack."""

from types import SimpleNamespace
import unittest

from utils.acf_relation_contract import resolve_acf_source


class ACFSourceContractTests(unittest.TestCase):
    def test_canonical_plan_key_precedes_historical_alias(self):
        self.assertEqual(
            resolve_acf_source({
                "autocon_acf_source": "RAW",
                "AutoCon_acf_source": "residual",
            }),
            "raw",
        )
        self.assertEqual(
            resolve_acf_source(SimpleNamespace(
                autocon_acf_source="raw",
                AutoCon_acf_source="trend",
            )),
            "raw",
        )

    def test_historical_alias_and_default_remain_compatible(self):
        self.assertEqual(
            resolve_acf_source({"AutoCon_acf_source": "Trend"}), "trend"
        )
        self.assertEqual(resolve_acf_source({}), "residual")


if __name__ == "__main__":
    unittest.main()
