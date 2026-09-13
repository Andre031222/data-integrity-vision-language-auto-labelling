#!/bin/bash
# AlpacaVision AI -- Setup completo del entorno

set -e
echo "AlpacaVision AI -- Setup"
echo "========================"

# Verificar Python
python --version || { echo "ERROR: Python no encontrado"; exit 1; }

# Crear entorno virtual
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python -m venv venv
fi

# Activar entorno
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

echo "Instalando dependencias..."
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt

echo ""
echo "Setup completado."
echo "Para activar el entorno:"
echo "  Windows: venv\\Scripts\\activate"
echo "  Linux:   source venv/bin/activate"
echo ""
echo "Próximos pasos:"
echo "  1. Copiar .env.example a .env y completar credenciales"
echo "  2. python src/data/download_roboflow.py"
echo "  3. python src/data/download_inaturalist.py --max_images 3000"
