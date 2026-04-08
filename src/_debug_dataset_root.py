import importlib.util
spec = importlib.util.spec_from_file_location('mod','src/01_explore_dataset.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
print('DATASET_ROOT =', repr(mod.DATASET_ROOT))
print('OUTPUT_DIR =', repr(mod.OUTPUT_DIR))
print('CLASS_MAP keys =', list(mod.CLASS_MAP.keys()))
