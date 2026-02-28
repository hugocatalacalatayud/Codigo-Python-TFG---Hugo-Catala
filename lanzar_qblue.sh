#!/bin/bash
#SBATCH --job-name=QML_Validacion
#SBATCH --output=qml_res_%j.out
#SBATCH --error=qml_err_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --account=ific60
#SBATCH --qos=debug
#SBATCH --partition=main

# NOTA: Cambia 'main' por 'quantum' o la partición que te hayan indicado si falla.
# 'debug' es para pruebas rápidas. Para el paper final usa 'gp_bscls' o similar.

# 1. Limpiar módulos y cargar Python
module purge
module load python/3.10
# Si existe un modulo específico de quantum:
# module load qiskit 

# 2. Activar tu entorno virtual (IMPORTANTE: Cambia esta ruta)
# Debes haber creado un entorno antes con: python -m venv mi_entorno
echo "Activando entorno virtual..."
source /home/ific/ific73/ific732102/mi_entorno_qml/bin/activate

# 3. Ejecutar el script
echo "Iniciando script en el nodo: $(hostname)"
python validacion_qblue.py

echo "Trabajo finalizado."