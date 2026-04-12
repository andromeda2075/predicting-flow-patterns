#!/usr/bin/env bash
set -euo pipefail

MOUNTPOINT="$HOME/carpeta_servidor"
REMOTE="sade06cy@zuse.math.uni-magdeburg.de:/home/sade06cy/predicting-flow-patterns/New_Experiment"
REMOTE_TARGET_DIR="/home/sade06cy/predicting-flow-patterns/New_Experiment"
SENTINEL="$HOME/.cache/predicting-flow-patterns/mount_done"

mkdir -p "$(dirname "$SENTINEL")"

if [ -f "$SENTINEL" ]; then
  echo "Operación ya ejecutada anteriormente. Saliendo."
  exit 0
fi

if ! command -v sshfs >/dev/null 2>&1; then
  echo "sshfs no está instalado. Instala sshfs y vuelve a intentarlo."
  exit 1
fi

# 1. Desmontar (si está montado)
if mountpoint -q "$MOUNTPOINT"; then
  echo "Desmontando $MOUNTPOINT..."
  sudo umount -l "$MOUNTPOINT"
fi

# 2. Montar
mkdir -p "$MOUNTPOINT"
echo "Montando $REMOTE -> $MOUNTPOINT (puede pedir contraseña o usar clave SSH)..."
sshfs "$REMOTE" "$MOUNTPOINT"

# 3. Vincular config
echo "Creando enlace simbólico para config.yaml..."
sudo ln -sf "$MOUNTPOINT/config.yaml" "$REMOTE_TARGET_DIR/config"

# 4. Vincular subcarpeta Results_training
echo "Creando enlace simbólico para Results_training..."
sudo ln -sfn "$MOUNTPOINT/Results_training" "$REMOTE_TARGET_DIR/Results_training"

# Marcar como ejecutado
touch "$SENTINEL"
echo "Operación completada. Sentinel creado en $SENTINEL"

exit 0