from jinja2 import Template

# ==========================================
# 1. テンプレートの定義 (本来は .cu.j2 ファイル)
# ==========================================
# {{ 変数名 }} の部分が、後からPythonによって置き換えられます。
template_str = """
extern "C"
__global__ void update_neuron_state(
    float* V,
    float* I_input,
    unsigned char* raster
) {
    // グローバルスレッドIDの計算
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < {{ num_neurons }}) {
        float v = V[tid];
        float I = I_input[tid];

        // --- モデル専用の数式 (ここが動的に変わる) ---
        {{ neuron_math }}
        // ----------------------------------------------

        // スパイク判定
        if (v > {{ threshold }}) {
            raster[tid] = 1;
            v = {{ reset_v }};
        } else {
            raster[tid] = 0;
        }

        V[tid] = v;
    }
}
"""

# Jinja2のテンプレートオブジェクトを作成
template = Template(template_str)


# ==========================================
# 2. モデルの定義 (本来は lif.py などのクラス内)
# ==========================================
# 例として、シンプルなLIFモデルの数式（C++のコード片）を用意します
lif_math_code = """
        // LIFモデルの計算
        float dt = 0.1f;
        float tau = 20.0f;
        float dv = (-(v - (-65.0f)) + I) * (dt / tau);
        v = v + dv;
"""


# ==========================================
# 3. レンダリング (本来は simulator.py 内)
# ==========================================
# テンプレートの「穴」に、変数や数式コードを流し込みます
generated_cuda_code = template.render(
    num_neurons=1024,             # ニューロン数
    neuron_math=lif_math_code.strip(), # 数式の文字列を注入
    threshold="30.0f",            # 閾値
    reset_v="-65.0f"              # リセット電位
)

# ==========================================
# 4. 結果の確認
# ==========================================
print("↓↓↓ 生成されたCUDAコード ↓↓↓\n")
print(generated_cuda_code)
print("\n↑↑↑ これを PyCUDA に渡してコンパイルします ↑↑↑")
