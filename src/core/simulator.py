# src/core/simulator.py
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from jinja2 import Environment, FileSystemLoader
import numpy as np

class Simulator:
    def __init__(self, config: dict, network_data: dict, neuron_model):
        self.config = config
        self.num_neurons = config["network"]["num_neurons"]
        self.network_data = network_data
        self.neuron_model = neuron_model
        
        # 1. 動的状態変数のGPUメモリ確保
        self.gpu_states = {}
        for key, np_array in self.neuron_model.get_initial_states().items():
            self.gpu_states[key] = gpuarray.to_gpu(np_array)

        self.I_d = gpuarray.zeros(self.num_neurons, dtype=np.float32)
        self.raster_d = gpuarray.zeros(self.num_neurons, dtype=np.uint8)

        # 2. カーネルのコンパイル
        self._compile_kernel()

        # 3. コンスタントメモリの転送
        self._transfer_constant_memory()

    def _compile_kernel(self):
        """Jinja2でCUDAコードを生成し、PyCUDAでコンパイルする"""
        # Jinja2環境のセットアップ
        env = Environment(loader=FileSystemLoader("src/templates"))
        template = env.get_template("base_kernel.cu.j2")
        comp = self.neuron_model.get_cuda_components()

        # テンプレートに変数と数式を流し込んで、完成版のCUDAコードを作る
        cuda_code = template.render(
            num_neurons=self.num_neurons,
            device_funcs=comp["device_funcs"],
            args=comp["args"],
            load_states=comp["load_states"],
            dynamics=comp["dynamics"],
            spike_logic=comp["spike_logic"],
            save_states=comp["save_states"]
        )

        # print("--- Generated CUDA Code ---")
        # print(cuda_code) # デバッグ用：完成したコードを確認できます
        # print("---------------------------")
        debug_file_path = "generated_kernel.cu"
        with open(debug_file_path, "w", encoding="utf-8") as f:
            f.write(cuda_code)
        print(f"Debug: Generated CUDA code saved to {debug_file_path}")

        # PyCUDAでコンパイル
        self.module = SourceModule(cuda_code)
        self.update_kernel = self.module.get_function("update_neuron_state")

    def _transfer_constant_memory(self):
        """モデルから要求されたコンスタントメモリをGPUに書き込む"""
        const_mem_dict = self.neuron_model.get_constant_memory()
        
        for symbol_name, np_array in const_mem_dict.items():
            # コンパイル済みのモジュールから、変数名(シンボル)のメモリアドレスを取得
            symbol_ptr, _ = self.module.get_global(symbol_name)
            
            # ホスト(CPU)のNumpy配列から、デバイス(GPU)のメモリアドレスへ直接コピー
            cuda.memcpy_htod(symbol_ptr, np_array)
            print(f"Transferred {symbol_name} to constant memory. Shape: {np_array.shape}")

    def set_input_current(self, I_array: np.ndarray):
        """外部から入力電流をセットする"""
        cuda.memcpy_htod(self.I_d.gpudata, I_array.astype(np.float32))

    def run(self, steps: int, I_history: np.ndarray = None):
        """シミュレーションのメインループ"""
        block = (256, 1, 1)
        grid = ((self.num_neurons + 255) // 256, 1)

        # GPUカーネルに渡す引数のリストを動的に構築
        # Jinja2の {{ args }} で定義した順番と同じ順番で変数を渡す必要があります
        kernel_args = [self.gpu_states[k] for k, _ in self.neuron_model.get_initial_states().items()]
        kernel_args += [self.I_d, self.raster_d]
        
        v_history = []
        raster_history = []
        
        for t in range(steps):
            if I_history is not None:
                cuda.memcpy_htod(self.I_d.gpudata, I_history[t].astype(np.float32))
                
            self.update_kernel(*kernel_args, block=block, grid=grid)
            
            # テスト用：Vの値を毎ステップ回収 (固定小数点なのでfloatに戻す)
            # P(1)は BIT_WIDTH_FRACTIONAL で、通常は10
            v_int = self.gpu_states["Vs_d"].get()
            v_float = v_int / (2 ** 10) 
            v_history.append(v_float)
            raster_history.append(self.raster_d.get().copy())
            
        return np.array(v_history), np.array(raster_history)