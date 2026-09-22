"""YAML を読む経路の不変条件。

- 重複キーを黙って後勝ちにしないこと
- リポジトリに実在する config / コンポーネント / task プロファイルがその規則を満たすこと

重複キーが通ると、プロファイルを 1 つ足したつもりが既存の定義を消していても落ちず、
**別のプロトコルで走った run** ができあがる。記録を見ても何で走ったか分からなくなる。
"""
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

root_path = Path(__file__).resolve().parents[1]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.core.config_manager import ConfigManager, load_yaml


def _write(tmp_dir, text: str) -> Path:
    path = Path(tmp_dir) / "profiles.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class DuplicateKeyTest(unittest.TestCase):
    """PyYAML の既定 (後勝ち) を上書きしていること。"""

    def test_duplicate_top_level_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "develop:\n  duration: 1.0\n\ndevelop:\n  duration: 2.0\n")
            with self.assertRaises(yaml.YAMLError) as caught:
                load_yaml(path)
            message = str(caught.exception)
            self.assertIn("develop", message)
            self.assertIn("1 行目", message)   # 捨てられるほうの位置も示すこと
            self.assertIn("4 行目", message)

    def test_duplicate_nested_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "develop:\n  record_hours: [0]\n  record_hours: [0, 12]\n")
            with self.assertRaises(yaml.YAMLError):
                load_yaml(path)

    def test_the_same_key_in_different_mappings_is_fine(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "a:\n  duration: 1.0\nb:\n  duration: 2.0\n")
            self.assertEqual(load_yaml(path), {"a": {"duration": 1.0}, "b": {"duration": 2.0}})

    def test_empty_file_is_an_empty_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(load_yaml(_write(tmp, "")), {})

    def test_config_manager_reads_through_the_same_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write(tmp, "a: 1\na: 2\n")
            with self.assertRaises(yaml.YAMLError):
                ConfigManager()._load_yaml(path)

    def test_a_missing_file_is_still_an_empty_mapping(self):
        # 規則を足しても「無いファイルは空扱い」は変えない (コンポーネント YAML は任意)
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(ConfigManager()._load_yaml(Path(tmp) / "absent.yaml"), {})


class RepositoryYamlTest(unittest.TestCase):
    """実在する YAML が規則を満たすこと。"""

    def test_every_config_and_profile_loads(self):
        targets = sorted(
            path
            for directory in ("configs", "scripts", "test")
            for path in (root_path / directory).rglob("*.yaml")
        )
        self.assertGreater(len(targets), 0)
        for path in targets:
            with self.subTest(path=str(path.relative_to(root_path))):
                load_yaml(path)


if __name__ == "__main__":
    unittest.main()
