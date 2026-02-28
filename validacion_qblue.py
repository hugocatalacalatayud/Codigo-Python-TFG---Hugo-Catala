import pennylane as qml
from pennylane import numpy as np
import json
import time
import os

# Intentamos importar Qiskit para ver los backends disponibles en el cluster
try:
    from qiskit_ibm_provider import IBMProvider
    from qiskit import Aer
    HAVE_QISKIT = True
except ImportError:
    HAVE_QISKIT = False

# ==============================================================================
# 1. PARÁMETROS APRENDIDOS (Tus resultados del entrenamiento)
# ==============================================================================
LEARNED_PARAMS = {
    "dt": 0.441527,
    "u_ff": 2.413383,
    "bias": 0.048092
}

# ==============================================================================
# 2. CONFIGURACIÓN DEL BACKEND (CLUSTER QBLUE)
# ==============================================================================
print("--- CONFIGURANDO ENTORNO QBLUE ---")

if HAVE_QISKIT:
    print("-> Librería Qiskit detectada.")
    # Si tienes un provider configurado en el entorno del BSC:
    try:
        # Esto listará los chips reales disponibles para tu cuenta
        provider = IBMProvider() 
        print("Backends disponibles:", [b.name for b in provider.backends()])
        
        # --- OPCIÓN A: HARDWARE REAL (Descomentar si sabes el nombre exacto) ---
        # backend_name = 'ibm_brisbane' # O el nombre que salga en la lista anterior
        # dev = qml.device('qiskit.ibmq', wires=4, backend=backend_name, provider=provider)
        
        # --- OPCIÓN B: SIMULACIÓN HPC (Por defecto para primera prueba) ---
        # Usamos el simulador Aer que corre rapidísimo en los nodos del BSC
        dev = qml.device("qiskit.aer", wires=4, shots=1024)
        print("-> Usando Backend: Qiskit Aer Simulator (HPC)")
        
    except Exception as e:
        print(f"-> Nota: No se pudo conectar al Provider IBMQ automáticamente ({e}).")
        print("-> Usando simulador local default.qubit como fallback.")
        dev = qml.device("default.qubit", wires=4)
else:
    print("-> Qiskit no detectado. Usando simulador PennyLane.")
    dev = qml.device("default.qubit", wires=4)


# ==============================================================================
# 3. CIRCUITO (IDÉNTICO AL ENTRENADO)
# ==============================================================================
def trotter_step(dt, u_imp, u_ff):
    qml.MultiRZ(2 * u_imp * dt, wires=[1, 3])
    qml.MultiRZ(2 * u_imp * dt, wires=[2, 3])
    qml.MultiRZ(2 * u_ff * dt, wires=[1, 2])
    qml.RX(dt, wires=1)
    qml.RX(dt, wires=2)
    qml.RX(dt, wires=3)

@qml.qnode(dev)
def circuit_inference(u_imp_val, theta_val):
    dt = LEARNED_PARAMS["dt"]
    u_ff = LEARNED_PARAMS["u_ff"]

    # Fase 1: Baño (Fuera control)
    qml.RY(theta_val, wires=1)
    qml.CNOT(wires=[1, 2])
    
    # Fase 2: Ramsey Controlado
    qml.Hadamard(wires=0)
    N_STEPS = 4
    for _ in range(N_STEPS):
        qml.ctrl(trotter_step, control=0)(dt=dt, u_imp=u_imp_val, u_ff=u_ff)

    qml.Hadamard(wires=0)
    return qml.expval(qml.PauliZ(0))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ==============================================================================
# 4. EJECUCIÓN DE PUNTOS DE PRUEBA
# ==============================================================================
test_points = [
    {"label": "BEC_Deep", "u": 5.5, "th": 3.0},
    {"label": "BEC_Edge", "u": 4.0, "th": 2.8},
    {"label": "Crossover", "u": 3.0, "th": 2.2},
    {"label": "BCS_Edge", "u": 2.0, "th": 1.8},
    {"label": "BCS_Deep", "u": 1.0, "th": 1.5}
]

results_data = []
print("\n--- INICIANDO INFERENCIA ---")

for pt in test_points:
    print(f"Procesando: {pt['label']}...")
    
    try:
        raw_signal = circuit_inference(pt["u"], pt["th"])
        # Post-procesado
        logits = raw_signal + LEARNED_PARAMS["bias"]
        prob_bcs = sigmoid(logits)
        pred_class = "BCS" if prob_bcs > 0.5 else "BEC"
        
        results_data.append({
            "label": pt["label"],
            "u_imp": float(pt["u"]),
            "theta": float(pt["th"]),
            "raw_signal": float(raw_signal),
            "prob_bcs": float(prob_bcs),
            "prediction": pred_class
        })
    except Exception as e:
        print(f"Error en punto {pt['label']}: {e}")

# ==============================================================================
# 5. GUARDAR RESULTADOS
# ==============================================================================
output_filename = "resultados_qblue_ific60.json"
with open(output_filename, "w") as f:
    json.dump(results_data, f, indent=4)

print(f"\nExito. Resultados guardados en: {output_filename}")