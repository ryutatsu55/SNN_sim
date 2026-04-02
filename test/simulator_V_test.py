import numpy as np
from pygenn import genn_model
from src.models.neurons.genn_pqn import get_pqn_3var_class, get_genn_pqn_params
from tqdm import tqdm

class Simulator:
    def __init__(self, cfg: dict):
        self.dt = cfg.get("dt", 0.1) # [ms]
        self.model = genn_model.GeNNModel("float", "pqn_single_test")
        self.model.dt = self.dt
        
        self.N = 1 # テスト用なので1ニューロン
        self._build_test_model(cfg)

    def _build_test_model(self, cfg):
        # 1. パラメータの取得 (ハードコード排除)
        # YAML設定から指定されたモード(例:"RSexci")と、I0などの上書きパラメータを取得
        mode = cfg.get("mode", "RSexci")
        
        pqn_params, pqn_init = get_genn_pqn_params(mode)

        # 2. ニューロングループの追加
        self.pop = self.model.add_neuron_population(
            "TestNeuron", self.N, get_pqn_3var_class(), pqn_params, pqn_init
        )

        
        print(f"[Simulator] Compiling model for {mode}...")
        self.model.build()
        self.model.load()

    def run_test(self, tmax_s: float, input_current: np.array):
        steps = int((tmax_s * 1000.0) / self.dt)
        print(f"[Simulator] Running {steps} steps with I_ext = {input_current}...")

        # テストとして、Isyn変数に直接一定の電流値を流し込む
        # (本来はシナプス入力だが、テスト用途で外部入力として悪用する)
        self.pop.vars["Iext"].view[:] = 0.0
        self.pop.vars["Iext"].push_to_device()

        v_trace = []
        
        # シミュレーション実行
        for step in tqdm(range(steps)):
            self.pop.vars["Iext"].view[:] = input_current[step]
            self.pop.vars["Iext"].push_to_device()

            self.model.step_time()
            
            # 毎ステップ、変数オブジェクト自身に pull させる (GeNN 5方式)
            self.pop.vars["V"].pull_from_device()
            
            v_trace.append(self.pop.vars["V"].view[0])

        return np.array(v_trace)