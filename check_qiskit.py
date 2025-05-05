# check_qiskit.py
import sys
print(f"Python path: {sys.path}")
try:
    import qiskit
    print(f"Qiskit found! Version: {qiskit.__version__}")
except ImportError as e:
    print(f"Failed to import Qiskit: {e}")