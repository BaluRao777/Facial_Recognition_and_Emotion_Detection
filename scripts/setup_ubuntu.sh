#!/usr/bin/env bash
# Setup on Ubuntu 24.04 LTS with NVIDIA RTX GPU (CUDA via TensorFlow pip extras)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== Facial Recognition + Emotion Detection — Ubuntu setup ==="

if ! command -v python3 &>/dev/null; then
  echo "Install Python 3.10+ first: sudo apt install python3 python3-venv python3-pip"
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip wheel
pip install -r requirements.txt

echo ""
echo "=== Running full project pipeline ==="
python main.py

echo ""
echo "=== Done ==="
echo "The project was started automatically in this setup script."
echo "To run manually later:"
echo "  source .venv/bin/activate"
echo "  python main.py"
echo "  python main.py --task emotion"
echo "  python main.py --task face"
echo "  python main.py --task eval"
