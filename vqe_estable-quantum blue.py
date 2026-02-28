#!/usr/bin/env python3
"""
VQE con Qiskit 1.0+ - Hamiltoniano Ising 1D
Versión FUNCIONANDO CON LIBRERÍAS ESTABLES
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("VQE - VARIATIONAL QUANTUM EIGENSOLVER (Qiskit 1.0+)")
print("=" * 80)

# ============================================================================
# VERIFICACIÓN E INSTALACIÓN DE LIBRERÍAS
# ============================================================================

def check_and_install():
    """Verifica e instala las librerías necesarias"""
    required = {
        'qiskit': 'qiskit',
        'qiskit_algorithms': 'qiskit-algorithms',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {module:20s} instalado")
        except ImportError:
            missing.append(package)
            print(f"✗ {module:20s} FALTA")
    
    if missing:
        print("\n⚠️  Instalando paquetes faltantes...")
        import subprocess
        for package in missing:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print("✓ Paquetes instalados correctamente\n")

print("\n🔧 Verificando dependencias...")
check_and_install()

# ============================================================================
# IMPORTACIONES FINALES
# ============================================================================

try:
    from qiskit.quantum_info import SparsePauliOp
    from qiskit.circuit.library import NLocal
    from qiskit.primitives import Estimator
    from qiskit_algorithms.minimum_eigensolvers import VQE
    from qiskit_algorithms.optimizers import COBYLA
    print("✓ Todas las importaciones de Qiskit cargadas correctamente\n")
except ImportError as e:
    print(f"❌ Error de importación: {e}")
    sys.exit(1)

# ============================================================================
# 1. DEFINICIÓN DEL HAMILTONIANO (H = -ZZ - XI - IX)
# ============================================================================

print("=" * 80)
print("1. DEFINICIÓN DEL HAMILTONIANO (Ising 1D - N=2)")
print("=" * 80)

# H = -1 * (Z ⊗ Z) - 1 * (X ⊗ I) - 1 * (I ⊗ X)
pauli_list = [
    ("ZZ", -1.0),
    ("XI", -1.0),
    ("IX", -1.0),
]

try:
    hamiltonian = SparsePauliOp.from_list(pauli_list)
    print("✓ Hamiltoniano creado exitosamente")
except Exception as e:
    print(f"❌ Error al crear Hamiltoniano: {e}")
    sys.exit(1)

# Autovalor exacto para verificación
try:
    H_matrix = hamiltonian.to_matrix()
    exact_eigenvalues = np.linalg.eigvalsh(H_matrix)
    exact_result = exact_eigenvalues.min()
    print(f"✓ Autovalor exacto (diagonalización): {exact_result:.6f}")
    print(f"✓ Número de qubits: {hamiltonian.num_qubits}")
    print(f"✓ Dimensión del espacio de Hilbert: {H_matrix.shape[0]}")
except Exception as e:
    print(f"❌ Error calculando autovalor exacto: {e}")
    sys.exit(1)

# ============================================================================
# 2. CONFIGURACIÓN DEL VQE (ANSATZ, OPTIMIZADOR, ESTIMADOR)
# ============================================================================

print("\n" + "=" * 80)
print("2. CONFIGURACIÓN DEL VQE")
print("=" * 80)

# Almacenar resultados intermedios
intermediate_results = {
    'iters': [],
    'values': [],
    'std': []
}

def callback_vqe(eval_count, parameters, mean, std):
    """Callback para almacenar energía en cada iteración"""
    intermediate_results['iters'].append(eval_count)
    intermediate_results['values'].append(mean)
    intermediate_results['std'].append(std)
    if eval_count % 10 == 0 or eval_count == 1:
        print(f"  Iter {eval_count:4d} | E = {mean:.8f} ± {std:.2e}")

# 2.1 Ansatz (Circuito Variacional)
try:
    num_qubits = hamiltonian.num_qubits
    
    ansatz = NLocal(
        num_qubits=num_qubits,
        rotation_blocks=['ry', 'rz'],      # Rotaciones locales (RY, RZ)
        entangler_blocks='cz',              # Puertas de entrelazamiento (CZ)
        reps=2,                             # 2 repeticiones del patrón
        entanglement='linear'               # Entrelazamiento lineal
    )
    
    print("\n2.1. ANSATZ (Circuito Variacional NLocal):")
    print(f"    ├─ Qubits: {num_qubits}")
    print(f"    ├─ Rotaciones: RY, RZ")
    print(f"    ├─ Entrelazamiento: CZ (topología lineal)")
    print(f"    ├─ Repeticiones: 2")
    print(f"    └─ Parámetros totales: {ansatz.num_parameters}")
    
except Exception as e:
    print(f"❌ Error creando Ansatz: {e}")
    sys.exit(1)

# 2.2 Optimizador (COBYLA)
try:
    optimizer = COBYLA(
        maxiter=200,           # Iteraciones máximas
        rhobeg=1.0,           # Tamaño inicial del simplex
        tol=1e-6              # Tolerancia
    )
    
    print("\n2.2. OPTIMIZADOR (COBYLA):")
    print(f"    ├─ Método: Constrained Optimization by Linear Approximation")
    print(f"    ├─ Iteraciones máx: 200")
    print(f"    ├─ Tolerancia: 1e-6")
    print(f"    └─ Status: Configurado ✓")
    
except Exception as e:
    print(f"❌ Error creando Optimizador: {e}")
    sys.exit(1)

# 2.3 Estimador (Primitivas de Qiskit)
try:
    estimator = Estimator()
    
    print("\n2.3. ESTIMADOR (Qiskit Primitives):")
    print(f"    ├─ Tipo: Estimator estándar")
    print(f"    ├─ Modo: Simulación exacta")
    print(f"    └─ Status: Inicializado ✓")
    
except Exception as e:
    print(f"❌ Error inicializando Estimador: {e}")
    print("    Intentando alternativa...")
    try:
        from qiskit_aer.primitives import Estimator as AerEstimator
        estimator = AerEstimator()
        print("    ✓ AerEstimator cargado como alternativa")
    except:
        print("❌ No se pudo inicializar estimador")
        sys.exit(1)

# ============================================================================
# 3. EJECUCIÓN DEL VQE
# ============================================================================

print("\n" + "=" * 80)
print("3. EJECUCIÓN DEL VQE")
print("=" * 80)
print("\nIteraciones del optimizador:\n")

try:
    vqe = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback_vqe
    )
    
    print("  Iniciando optimización...")
    vqe_result = vqe.compute_minimum_eigenvalue(hamiltonian)
    print("\n  ✓ Optimización completada exitosamente")
    
except Exception as e:
    print(f"\n❌ Error durante VQE: {e}")
    print("   Esto puede ocurrir por versiones incompatibles de Qiskit")
    print("   Intenta reinstalar: pip install qiskit qiskit-algorithms -U")
    sys.exit(1)

# ============================================================================
# 4. ANÁLISIS DE RESULTADOS
# ============================================================================

print("\n" + "=" * 80)
print("4. RESULTADOS FINALES")
print("=" * 80)

try:
    vqe_eigenvalue = vqe_result.eigenvalue.real
    error_abs = abs(vqe_eigenvalue - exact_result)
    error_rel = (error_abs / abs(exact_result)) * 100 if exact_result != 0 else 0
    
    # Mejora del error
    initial_error = abs(intermediate_results['values'][0] - exact_result)
    improvement = ((initial_error - error_abs) / initial_error) * 100 if initial_error > 0 else 0
    
    print(f"\n📊 Comparación de Resultados:")
    print(f"   ├─ Autovalor VQE:          {vqe_eigenvalue:12.8f} Ha")
    print(f"   ├─ Autovalor Exacto:       {exact_result:12.8f} Ha")
    print(f"   ├─ Error Absoluto:         {error_abs:12.2e} Ha")
    print(f"   ├─ Error Relativo:         {error_rel:12.4f} %")
    print(f"   └─ Mejora Total:           {improvement:12.2f} %")
    
    print(f"\n📈 Estadísticas de Optimización:")
    print(f"   ├─ Iteraciones totales:    {len(intermediate_results['iters']):d}")
    print(f"   ├─ Energía inicial:        {intermediate_results['values'][0]:12.8f} Ha")
    print(f"   ├─ Energía final:          {intermediate_results['values'][-1]:12.8f} Ha")
    print(f"   └─ Desv. estándar final:   {intermediate_results['std'][-1]:12.2e}")
    
except Exception as e:
    print(f"❌ Error procesando resultados: {e}")
    sys.exit(1)

# ============================================================================
# 5. VISUALIZACIÓN
# ============================================================================

print("\n" + "=" * 80)
print("5. GENERANDO VISUALIZACIONES")
print("=" * 80)

try:
    fig = plt.figure(figsize=(16, 6))
    
    # Gráfica 1: Convergencia
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(intermediate_results['iters'], intermediate_results['values'],
             'o-', color='#0066cc', linewidth=2.5, markersize=6, 
             alpha=0.8, label='Energía VQE')
    ax1.axhline(y=exact_result, color='#ff6633', linestyle='--',
                linewidth=2.5, label=f'Energía Exacta: {exact_result:.6f}')
    ax1.fill_between(intermediate_results['iters'],
                     intermediate_results['values'],
                     exact_result, alpha=0.2, color='green',
                     label='Diferencia')
    
    ax1.set_xlabel('Evaluaciones del Circuito', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Energía $\\langle H \\rangle$ (Ha)', fontsize=11, fontweight='bold')
    ax1.set_title('Convergencia del VQE', fontsize=12, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=9, loc='best', framealpha=0.95)
    
    # Gráfica 2: Error (escala log)
    ax2 = plt.subplot(1, 3, 2)
    errors = np.abs(np.array(intermediate_results['values']) - exact_result)
    errors = np.maximum(errors, 1e-12)  # Evitar log(0)
    
    ax2.semilogy(intermediate_results['iters'], errors,
                 's-', color='#cc0033', linewidth=2.5, markersize=5,
                 alpha=0.8, label='Error Absoluto')
    ax2.axhline(y=1e-6, color='green', linestyle='--', linewidth=2,
                label='Tolerancia (1e-6)', alpha=0.7)
    
    ax2.set_xlabel('Evaluaciones del Circuito', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Error $|E_{VQE} - E_{exact}|$ (Ha)', fontsize=11, fontweight='bold')
    ax2.set_title('Error de Aproximación (escala log)', fontsize=12, fontweight='bold')
    ax2.grid(True, which='both', linestyle=':', alpha=0.6)
    ax2.legend(fontsize=9, loc='best', framealpha=0.95)
    
    # Gráfica 3: Desviación estándar
    ax3 = plt.subplot(1, 3, 3)
    ax3.semilogy(intermediate_results['iters'], intermediate_results['std'],
                 '^-', color='#ff9900', linewidth=2.5, markersize=5,
                 alpha=0.8, label='Desv. Estándar')
    
    ax3.set_xlabel('Evaluaciones del Circuito', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Desv. Estándar (Ha)', fontsize=11, fontweight='bold')
    ax3.set_title('Incertidumbre de Medición', fontsize=12, fontweight='bold')
    ax3.grid(True, which='both', linestyle=':', alpha=0.6)
    ax3.legend(fontsize=9, loc='best', framealpha=0.95)
    
    plt.tight_layout()
    filename = 'vqe_resultados.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráficas guardadas en: {filename}")
    
    plt.show()
    
except Exception as e:
    print(f"⚠️  Error generando gráficas: {e}")
    print("   (Las gráficas son opcionales, los resultados se calcularon correctamente)")

# ============================================================================
# 6. RESUMEN FINAL
# ============================================================================

print("\n" + "=" * 80)
print("✅ EJECUCIÓN COMPLETADA CON ÉXITO")
print("=" * 80)

print("\n📋 RESUMEN:")
print(f"   • Problema: Hamiltoniano Ising 1D (2 qubits)")
print(f"   • Algoritmo: VQE con Ansatz NLocal")
print(f"   • Optimizador: COBYLA ({len(intermediate_results['iters'])} iteraciones)")
print(f"   • Precisión lograda: {error_rel:.4f}% de error relativo")
print(f"   • Estado: EXITOSO ✓")

print("\n" + "=" * 80 + "\n")
