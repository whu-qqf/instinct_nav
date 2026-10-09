import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


spec = importlib.util.spec_from_file_location(
    "live_episode_diagnostic", Path(__file__).parents[1] / "scripts/live_episode_diagnostic.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Query:
    def __init__(self, hits):
        self.hits = hits

    def raycast_all(self, origin, direction, distance, callback):
        for hit in self.hits:
            if not callback(hit):
                break


class TerrainQueryTest(unittest.TestCase):
    def test_topmost_terrain_only(self):
        def hit(path, z):
            return SimpleNamespace(collision=path, rigid_body="", position=(1, 2, z))
        query = Query([hit("/World/Robot", 9), hit("/World/ground_extra", 8),
                       hit("/World/ground/mesh", 0), hit("/World/ground", 1)])
        self.assertEqual(module.terrain_query_height(query, (1, 2, 20))["position"], [1, 2, 1])

    def test_missing_is_explicit(self):
        self.assertIsNone(module.terrain_query_height(Query([]), (0, 0, 20)))

    def test_bad_callback_is_not_a_missing_hit(self):
        with self.assertRaises(RuntimeError):
            module.terrain_query_height(Query([object()]), (0, 0, 20))


if __name__ == "__main__":
    unittest.main()
