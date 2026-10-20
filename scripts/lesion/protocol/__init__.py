"""**実験の進め方**を決める層。何を切るか、切った後に何を止めるか、親 run を正しく引き継げたか。

入口 (`__main__.py` / `run_one.py` / `replot.py`) だけが import する。`analysis/` / `store/` /
`report/` はここを import しない —— 指標の計算と記録の読み書きは、run がどう作られたかを
知らずに済むようにするため。GeNN も matplotlib も import しない。
"""
