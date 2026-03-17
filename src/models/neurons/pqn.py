import numpy as np
# 既存の PQNparam クラスを利用すると仮定
from src.models.neurons.PQN_origin import PQNengine as PQNparam
from src.core.base_models import BaseNeuron
import textwrap

class PQNNeuron(BaseNeuron):
    def __init__(self, num_neurons: int, neuron_types: np.ndarray):
        super().__init__(num_neurons)
        self.neuron_types = neuron_types
        
        # PQNparamの初期化
        self.models = [
            PQNparam(mode="RSexci"), 
            PQNparam(mode="RSinhi"), 
            PQNparam(mode="FS"), 
            PQNparam(mode="LTS"), 
            PQNparam(mode="IB"), 
            PQNparam(mode="EB"), 
            PQNparam(mode="PB")
        ]
        
        # AllParams の構築
        self.all_params_h = np.zeros((7, 34), dtype=np.int32)
        for i, model in enumerate(self.models):
            param = self._build_param_array(model)
            # LTS/IB以外はetaを1.0(シフト後)にしておくパッチ
            if param[31] == 0 and param[32] == 0:
                param[31] = 1 << param[0]
                param[32] = 1 << param[0]
            self.all_params_h[i] = param

    def _build_param_array(self, PQN):
        """PQN.py のパラメータから34要素の配列を作る（元の _param_h_init と同じ）"""
        param = np.zeros(34, dtype=np.int32)
        if PQN.mode in ["RSexci", "RSinhi", "FS", "EB", "LTS", "IB", "PB"]:
            param[0] = PQN.BIT_Y_SHIFT
            param[1] = PQN.BIT_WIDTH_FRACTIONAL
            param[2] = PQN.Y.get("v_vv_S", 0)
            param[3] = PQN.Y.get("v_v_S", 0)
            param[4] = PQN.Y.get("v_c_S", 0)
            param[5] = PQN.Y.get("v_n", 0)
            param[6] = PQN.Y.get("v_q", 0)
            param[7] = PQN.Y.get("v_I", 0)
            param[8] = PQN.Y.get("v_vv_L", 0)
            param[9] = PQN.Y.get("v_v_L", 0)
            param[10] = PQN.Y.get("v_c_L", 0)
            param[11] = PQN.Y.get("rg", 0)
            param[12] = PQN.Y.get("n_vv_S", 0)
            param[13] = PQN.Y.get("n_v_S", 0)
            param[14] = PQN.Y.get("n_c_S", 0)
            param[15] = PQN.Y.get("n_n", 0)
            param[16] = PQN.Y.get("n_vv_L", 0)
            param[17] = PQN.Y.get("n_v_L", 0)
            param[18] = PQN.Y.get("n_c_L", 0)
            param[19] = PQN.Y.get("rh", 0)
            param[20] = PQN.Y.get("q_vv_S", 0)
            param[21] = PQN.Y.get("q_v_S", 0)
            param[22] = PQN.Y.get("q_c_S", 0)
            param[23] = PQN.Y.get("q_q", 0)
            param[24] = PQN.Y.get("q_vv_L", 0)
            param[25] = PQN.Y.get("q_v_L", 0)
            param[26] = PQN.Y.get("q_c_L", 0)
            param[27] = PQN.Y.get("u_v", 0)
            param[28] = PQN.Y.get("u_u", 0)
            param[29] = PQN.Y.get("u_c", 0)
            param[30] = PQN.Y.get("ru", 0)
            param[31] = PQN.Y.get("n_uS", 0)
            param[32] = PQN.Y.get("n_uL", 0)
            param[33] = PQN.Y.get("v_u", 0)
        return param

    def get_constant_memory(self) -> dict:
        # テンプレート側の __constant__ 宣言と同じ変数名をキーにする
        return {
            "AllParams": self.all_params_h
        }

    def get_initial_states(self) -> dict:
        """GPUに確保させる配列の初期値を辞書で返す"""
        init_vs = np.array([m.state_variable_v for m in self.models], dtype=np.int64)
        init_ns = np.array([m.state_variable_n for m in self.models], dtype=np.int64)
        init_qs = np.array([m.state_variable_q for m in self.models], dtype=np.int64)
        init_us = np.array([m.state_variable_u for m in self.models], dtype=np.int64)
        
        return {
            "Vs_d": init_vs[self.neuron_types],
            "Ns_d": init_ns[self.neuron_types],
            "Qs_d": init_qs[self.neuron_types],
            "Us_d": init_us[self.neuron_types],
            "neuron_type_d": self.neuron_types.astype(np.uint8),
            "last_spike_d": np.zeros(self.num_neurons, dtype=np.uint8)
        }

    def get_cuda_components(self) -> dict:
        """Jinja2テンプレートに流し込むCUDAコードの各パーツを返す"""
        
        # ヘルパー関数（デバイス関数）はカーネルの外に配置する必要があるため、別枠で定義
        device_funcs = """
        #define P(idx) p_base[idx]
        
        __device__ int64_t v0(int64_t v, int64_t n, int64_t q, int64_t u, int64_t I, int64_t vv, const int* p_base) {
            bool neg = (v < 0);
            int64_t c_vv = neg ? P(2) : P(8);
            int64_t c_v  = neg ? P(3) : P(9);
            int64_t c_c  = neg ? P(4) : P(10);
            return ((c_vv * vv) >> P(0)) + ((c_v  * v)  >> P(0)) + c_c +
                   ((P(5) * n) >> P(0)) + ((P(6) * q) >> P(0)) + ((P(7) * I) >> P(0)) - ((P(33) * u) >> P(0));
        }
        __device__ int64_t n0(int64_t v, int64_t n, int64_t u, int64_t vv, const int* p_base) {
            bool cond = (v < P(11));
            int64_t c_vv = cond ? P(12) : P(16);
            int64_t c_v  = cond ? P(13) : P(17);
            int64_t c_c  = cond ? P(14) : P(18);
            int64_t dn = ((c_vv * vv) >> P(0)) + ((c_v  * v)  >> P(0)) + c_c + ((P(15) * n) >> P(0));
            int64_t eta = (u < (int64_t)P(30)) ? (int64_t)P(31) : (int64_t)P(32);
            return (dn * eta) >> P(0);
        }
        __device__ int64_t q0(int64_t v, int64_t q, int64_t vv, const int* p_base) {
            bool cond = (v < P(19));
            int64_t c_vv = cond ? P(20) : P(24);
            int64_t c_v  = cond ? P(21) : P(25);
            int64_t c_c  = cond ? P(22) : P(26);
            return ((c_vv * vv) >> P(0)) + ((c_v  * v)  >> P(0)) + c_c + ((P(23) * q) >> P(0));
        }
        __device__ int64_t u0(int64_t v, int64_t u, const int* p_base) {
            return (((int64_t)P(27) * v) >> P(0)) + (((int64_t)P(28) * u) >> P(0)) + (int64_t)P(29);
        }
        """

        # カーネルの引数
        args = """
        int64_t* Vs_d, 
        int64_t* Ns_d, 
        int64_t* Qs_d, 
        int64_t* Us_d, 
        const unsigned char* neuron_type_d,
        unsigned char* last_spike_d

        """
        # (※ AllParams は __constant__ メモリとして宣言するため、引数には含めず、Simulator側で別途コピーします)

        # 状態変数の読み込み
        load_states = """
        int64_t v = Vs_d[tid];
        int64_t n = Ns_d[tid];
        int64_t q = Qs_d[tid];
        int64_t u = Us_d[tid];
        
        int type_idx = neuron_type_d[tid];
        const int* p_base = AllParams[type_idx];
        
        // I_inputとsynaptic_inputの加算はテンプレート側で float I として渡される想定
        int64_t I_fixed = (int64_t)(I_float * (1 << P(1)));
        int64_t vv = (int64_t)((v * v) / (1LL << P(1)));
        """

        # ダイナミクス（更新式）
        dynamics = """
        int64_t dv = v0(v, n, q, u, I_fixed, vv, p_base);
        int64_t dn = n0(v, n, u, vv, p_base);
        int64_t dq = q0(v, q, vv, p_base);
        int64_t du = u0(v, u, p_base);
        """

        # スパイク判定とリセット（PQNはvのリセットをv0関数内のダイナミクスで処理するため、状態リセットは行わずフラグのみ立てる）
        spike_logic = """
        int64_t threshold = (4 << P(1));
        unsigned char current_spike = (v + dv > threshold) ? 1 : 0;
        
        // 前回発火しておらず、今回閾値を超えた瞬間だけ1にする (Pos-Edge)
        raster[tid] = (current_spike && !last_spike_d[tid]);
        
        // 状態を更新
        last_spike_d[tid] = current_spike;
        """

        # 状態変数の書き戻し
        save_states = """
        Vs_d[tid] = v + dv;
        Ns_d[tid] = n + dn;
        Qs_d[tid] = q + dq;
        Us_d[tid] = u + du;
        """

        return {
            "device_funcs": textwrap.dedent(device_funcs).strip(),
            "args": textwrap.dedent(args).strip(),
            "load_states": textwrap.dedent(load_states).strip(),
            "dynamics": textwrap.dedent(dynamics).strip(),
            "spike_logic": textwrap.dedent(spike_logic).strip(),
            "save_states": textwrap.dedent(save_states).strip()
        }
