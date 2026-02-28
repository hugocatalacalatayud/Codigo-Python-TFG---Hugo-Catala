import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# ==============================================================================
#  SIMULACIÓN POLARON DE FERMI: SCALING 6, 8, 10 QUBITS
# ==============================================================================

class PolaronSimulation:
    def __init__(self, n_qubits):
        """
        N_QUBITS: Total de Qubits (Ancilla + Baño + Impureza)
        Estructura: [ANCILLA, BATH_UP_1, BATH_DOWN_1, ..., IMPURITY]
        """
        self.n_qubits = n_qubits
        self.ancilla = 0
        self.impurity = n_qubits - 1
        self.bath_wires = list(range(1, n_qubits - 1))
        
        # Física
        self.EPS_UP = 0.5
        self.EPS_DOWN = 0.5
        self.EPS_IMP = 0.2
        self.U_FF = 2.0
        
        self.wires_list = [self.ancilla] + self.bath_wires + [self.impurity]
        self.dev = qml.device("default.qubit", wires=self.wires_list)

    def get_hamiltonian(self, u_imp_val):
        coeffs = []
        ops = []
        
        # 1. Impureza
        coeffs.append(-0.5 * self.EPS_IMP)
        ops.append(qml.PauliZ(self.impurity))
        
        # 2. Baño (Pares)
        for i in range(0, len(self.bath_wires), 2):
            w_up = self.bath_wires[i]
            w_down = self.bath_wires[i+1]
            
            # Kinematics
            coeffs.append(-0.5 * self.EPS_UP); ops.append(qml.PauliZ(w_up))
            coeffs.append(-0.5 * self.EPS_DOWN); ops.append(qml.PauliZ(w_down))
            
            # Bath-Bath Interaction (Pares de Cooper)
            c_int = 0.25 * self.U_FF
            coeffs.append(c_int); ops.append(qml.PauliZ(w_up) @ qml.PauliZ(w_down))
            coeffs.append(-c_int); ops.append(qml.PauliZ(w_up))
            coeffs.append(-c_int); ops.append(qml.PauliZ(w_down))
            
            # Polaron Interaction
            for w_bath in [w_up, w_down]:
                c_imp = 0.25 * u_imp_val
                coeffs.append(c_imp); ops.append(qml.PauliZ(self.impurity) @ qml.PauliZ(w_bath))
                coeffs.append(-c_imp); ops.append(qml.PauliZ(self.impurity))
                coeffs.append(-c_imp); ops.append(qml.PauliZ(w_bath))
                
        return coeffs, ops

    def run_circuit(self, time_val, u_imp_val, theta_bcs, steps=5):
        @qml.qnode(self.dev)
        def circuit():
            # Prep Baño BCS
            for i in range(0, len(self.bath_wires), 2):
                w_up = self.bath_wires[i]
                w_down = self.bath_wires[i+1]
                qml.RY(theta_bcs, wires=w_up)
                qml.CNOT(wires=[w_up, w_down])
            
            # Inicializar Impureza (Ocupado)
            qml.PauliX(wires=self.impurity)
            
            # Ramsey
            qml.Hadamard(wires=self.ancilla)
            
            if time_val > 0:
                dt = time_val / steps
                coeffs, ops = self.get_hamiltonian(u_imp_val)
                for _ in range(steps):
                    for c, op in zip(coeffs, ops):
                        angle = 2 * c * dt
                        if len(op.wires) == 1:
                            qml.ctrl(qml.RZ, control=self.ancilla)(angle, wires=op.wires)
                        else:
                            qml.ctrl(qml.MultiRZ, control=self.ancilla)(angle, wires=op.wires)
            
            qml.Hadamard(wires=self.ancilla)
            return qml.probs(wires=self.ancilla)
        
        return circuit()

if __name__ == "__main__":
    # CONFIGURACIÓN SOLICITADA
    SIZES = [6, 8, 10]
    
    # --- 1. Generación de Dinámica S(t) ---
    print("--- Generando Dinámica S(t) (6, 8, 10 Qubits) ---")
    dynamics_data = []
    t_vals = np.linspace(0, 5.0, 30)
    
    for n in SIZES:
        print(f"Simulando N={n}...")
        sim = PolaronSimulation(n)
        for t in t_vals:
            p = sim.run_circuit(t, u_imp_val=2.5, theta_bcs=1.57, steps=6)
            s_t = p[0] - p[1]
            dynamics_data.append({"N_Qubits": n, "Time": t, "S(t)": float(s_t)})
            
    df_dyn = pd.DataFrame(dynamics_data)
    df_dyn.to_csv("polaron_dynamics.csv", index=False)
    print("Guardado: polaron_dynamics.csv")

    # --- 2. Generación de Espectro (Barrido U) ---
    print("\n--- Generando Espectro (Heatmap Data) ---")
    spectrum_data = []
    u_vals = np.linspace(0.1, 6.0, 20)
    t_long = np.linspace(0, 10.0, 60)
    dt = t_long[1] - t_long[0]
    
    for n in SIZES:
        print(f"Calculando Espectro N={n}...")
        sim = PolaronSimulation(n)
        for u in u_vals:
            sig = []
            for t in t_long:
                p = sim.run_circuit(t, u_imp_val=u, theta_bcs=1.57, steps=4)
                sig.append(p[0] - p[1])
            
            fft_vals = np.abs(np.fft.fft(sig))
            freqs = np.fft.fftfreq(len(sig), d=dt)
            mask = freqs >= 0
            
            for f, amp in zip(freqs[mask], fft_vals[mask]):
                spectrum_data.append({
                    "N_Qubits": n, 
                    "U_imp": u, 
                    "Frequency": f, 
                    "Amplitude": amp
                })
                
    df_spec = pd.DataFrame(spectrum_data)
    df_spec.to_csv("polaron_spectrum.csv", index=False)
    print("Guardado: polaron_spectrum.csv")