#!/usr/bin/env bash
# Build llama-server (and llama-tokenize) of piquant's llama.cpp baseline for the Cerno 4B
# comparison (run inside WSL Ubuntu as root). Same source, tag and Pi flags as piquant's
# bench/llamacpp/build.sh (which clones $SRC), but in separate build directories, so the
# existing baseline builds are not touched:
#   host (x86, GGML_NATIVE=ON):   $SRC/build-host-server  -> <out>/host/llama-server/
#   Pi 4 (static aarch64, A72):   $SRC/build-pi4-server   -> <out>/aarch64/llama-server/
# No web UI, no OpenSSL (the server is only used on localhost).
#   bash scripts/build_llama_server.sh [tag, default v0.7.0] [out dir, default ./build]
set -euo pipefail

TAG="${1:-v0.7.0}"
OUT="${2:-$(cd "$(dirname "$0")/.." && pwd)/build}"
SRC=/root/src/llama.cpp-"$TAG"
OUT_PI="$OUT/aarch64/llama-server"
OUT_HOST="$OUT/host/llama-server"
JOBS="$(nproc)"
[ -d "$SRC" ] || { echo "missing $SRC (bench/llamacpp/build.sh clones it)"; exit 1; }

COMMON=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
        -DLLAMA_BUILD_SERVER=ON -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_UI=OFF
        -DLLAMA_USE_PREBUILT_UI=OFF -DBUILD_SHARED_LIBS=OFF)
TARGETS=(llama-server llama-tokenize)

echo "== host build"
cmake -S "$SRC" -B "$SRC/build-host-server" "${COMMON[@]}" > "$SRC/configure-host-server.log"
cmake --build "$SRC/build-host-server" -j "$JOBS" --target "${TARGETS[@]}" > "$SRC/build-host-server.log"

echo "== Pi 4 cross build"
PI_FLAGS="-mcpu=cortex-a72"
cmake -S "$SRC" -B "$SRC/build-pi4-server" "${COMMON[@]}" \
    -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
    -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
    -DCMAKE_C_FLAGS="$PI_FLAGS" -DCMAKE_CXX_FLAGS="$PI_FLAGS" \
    -DCMAKE_EXE_LINKER_FLAGS="-static" \
    -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=OFF -DGGML_CPU_ARM_ARCH=armv8-a -DGGML_OPENMP=ON \
    -DOpenMP_gomp_LIBRARY=/usr/lib/gcc-cross/aarch64-linux-gnu/15/libgomp.a \
    > "$SRC/configure-pi4-server.log"
cmake --build "$SRC/build-pi4-server" -j "$JOBS" --target "${TARGETS[@]}" > "$SRC/build-pi4-server.log"

mkdir -p "$OUT_PI" "$OUT_HOST"
cp "$SRC"/build-pi4-server/bin/{llama-server,llama-tokenize} "$OUT_PI"/
cp "$SRC"/build-host-server/bin/{llama-server,llama-tokenize} "$OUT_HOST"/
COMMIT="$(grep -rho 'LLAMA_COMMIT = "[^"]*"' "$SRC/build-host-server/common" 2>/dev/null | head -1)"
for d in "$OUT_PI" "$OUT_HOST"; do
    {
        echo "llama.cpp tag: $TAG"
        echo "commit: ${COMMIT:-unknown}"
        echo "built: $(date -Iseconds)"
        echo "cmake: ${COMMON[*]}"
        echo "pi flags: $PI_FLAGS, static, GGML_NATIVE=OFF, GGML_CPU_ARM_ARCH=armv8-a, OpenMP"
        echo "cross compiler: $(aarch64-linux-gnu-gcc --version | head -1)"
        echo "host compiler: $(gcc --version | head -1)"
        (cd "$d" && sha256sum -- llama-server llama-tokenize)
    } > "$d/BUILD_INFO.txt"
done
file "$OUT_PI/llama-server"
cat "$OUT_PI/BUILD_INFO.txt"
