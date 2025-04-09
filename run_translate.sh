#!/bin/bash
set -e

VENV_DIR="./venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
    echo "✅ Виртуальное окружение создано."
fi

source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Готово! Теперь можно запускать:"
echo "python translate_xml.py --path путь/к/папке"
