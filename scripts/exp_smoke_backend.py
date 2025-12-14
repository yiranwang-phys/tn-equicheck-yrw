import argparse
from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli

from qem_yrw_project.adapters.factory import make_backend

def obs_z0(statevector):
    # <Z0> on single-qubit state
    return statevector.expectation_value(Pauli("Z")).real

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="qiskit")
    args = ap.parse_args()

    backend = make_backend(args.backend)

    qc = QuantumCircuit(1)
    qc.h(0)  # |+> => <Z>=0

    val, meta = backend.ideal_expectation(qc, obs_z0)
    print("backend:", meta.backend, " <Z> =", val)

if __name__ == "__main__":
    main()
