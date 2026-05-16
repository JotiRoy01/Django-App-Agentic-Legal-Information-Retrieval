from pathlib import Path
#BASE_dir = 
print(f"Path(__file__) {Path(__file__)}")
print(f"Path(__file__).resolve() {Path(__file__).resolve()}")
print(f"Path(__file__).resolve().parent {Path(__file__).resolve().parent}")
print(f"Path(__file__).resolve().parent.parent {Path(__file__).resolve().parent.parent}")