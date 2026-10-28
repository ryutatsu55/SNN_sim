"""YAML を読む経路の不変条件。

- 重複キーを黙って後勝ちにしないこと
- リポジトリに実在する config / コンポーネント / task プロファイルがその規則を満たすこと
- メイン config に書いた値がプロファイルの値より優先され、書かなかった値はプロファイルから来ること
  (逆になると、掃引したつもりで全条件が同じ値で走る)

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


class ProfileOverlayTest(unittest.TestCase):
    """メイン config に書いた値がプロファイル (components/*.yaml) より優先されること。

    逆だと、条件ごとの config にパラメータを書いても黙ってプロファイルの値で走り、
    run の config.yaml にもプロファイルの値が記録される (掃引したつもりで全条件が同じになる)。
    """

    BASE = root_path / "test" / "experiments" / "no_space_100.yaml"

    def _resolve(self, edit):
        main = load_yaml(self.BASE)
        edit(main)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "main.yaml"
            path.write_text(yaml.safe_dump(main), encoding="utf-8")
            # 実験の task.yaml を借りない (借りると実験の都合でこのテストが動く)
            task = Path(tmp) / "task.yaml"
            task.write_text("probe: {}\n", encoding="utf-8")
            return ConfigManager().resolve(str(path), "probe", task_path=task)

    def test_inline_value_wins_over_profile(self):
        def edit(main):
            main["synapses"]["from_Exc"]["plasticity"]["tau_rec"] = 1234.0
            main["synapses"]["from_Exc"]["synapse"]["tau"] = 7.0
            main["neurons"]["Layer_Exc"]["f_rest"] = 0.1
        config = self._resolve(edit)
        self.assertEqual(config.synapses["from_Exc"].plasticity.tau_rec, 1234.0)
        self.assertEqual(config.synapses["from_Exc"].synapse.tau, 7.0)
        self.assertEqual(config.neurons["Layer_Exc"].f_rest, 0.1)

    def test_unwritten_values_come_from_profile(self):
        profile = load_yaml(root_path / "configs" / "components" / "plasticity.yaml")
        expected = profile["custom_Akita"]["e-stdp"]
        config = self._resolve(lambda main: None)
        plasticity = config.synapses["from_Exc"].plasticity
        for key in ("tau_rec", "U", "g_max"):
            self.assertEqual(getattr(plasticity, key), expected[key])
        # 上書きは書いたグループだけに効く
        other = self._resolve(
            lambda main: main["synapses"]["from_Exc"]["plasticity"].update(tau_rec=999.0))
        self.assertEqual(other.synapses["from_Inh"].plasticity.tau_rec,
                         profile["custom_Akita"]["i-stdp"]["tau_rec"])


class NetworkOverlayTest(ProfileOverlayTest):
    """network のコンポーネントも、メイン config の値がプロファイルより優先されること。"""

    def test_inline_value_wins_over_profile(self):
        profile = load_yaml(root_path / "configs" / "components" / "connections.yaml")
        main = load_yaml(self.BASE)
        name = main["network"]["connection"]
        expected = dict(profile[name])
        key = next(iter(k for k, v in expected.items() if isinstance(v, (int, float))
                        and not isinstance(v, bool)))

        def edit(m):
            m["network"]["connection"] = {"profile_name": name, key: 0.123}
        config = self._resolve(edit)
        connection = config.network.connection
        self.assertEqual(connection.profile_name, name)
        self.assertEqual(getattr(connection, key), 0.123)
        for k, v in expected.items():
            if k != key:
                self.assertEqual(getattr(connection, k), v)

    def test_unwritten_values_come_from_profile(self):
        main = load_yaml(self.BASE)
        name = main["network"]["connection"]
        as_name = self._resolve(lambda m: None).network.connection
        as_dict = self._resolve(
            lambda m: m["network"].update(connection={"profile_name": name})).network.connection
        self.assertEqual(as_name.model_dump(), as_dict.model_dump())


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
