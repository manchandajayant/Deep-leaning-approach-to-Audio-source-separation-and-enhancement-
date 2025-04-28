#!/usr/bin/env bash
set -euo pipefail

DATASET_URL="https://zenodo.org/record/1117372/files/musdb18.zip?download=1"

DEST_DIR="UNET/musdb18"

echo "→ Creating destination folder: ${DEST_DIR}"
mkdir -p "${DEST_DIR}"

echo "→ Downloading MUSDB18 dataset from Zenodo…"
wget -O musdb18.zip "${DATASET_URL}"

echo "→ Extracting archive…"

TMPDIR="$(mktemp -d)"
unzip -q musdb18.zip -d "${TMPDIR}"

echo "→ Moving files into ${DEST_DIR}"
mv "${TMPDIR}/musdb18/"* "${DEST_DIR}/"

echo "→ Removing temporary files…"
rm -rf "${TMPDIR}" musdb18.zip

echo "✅ MUSDB18 is now unpacked directly under ${DEST_DIR}"

