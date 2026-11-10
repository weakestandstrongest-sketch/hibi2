#!/usr/bin/env bash
# まいにち の AI（ai/）を作り直す。
# Personal AI Engine（weakestandstrongest-sketch/ai）を決まったコミットで取ってきて、その中に app/ を置いてビルドする。
# できた ai/ はそのままコミットする（GitHub Pages はビルドしないため）。
set -euo pipefail
ENGINE_REPO=https://github.com/weakestandstrongest-sketch/ai.git
ENGINE_REF=39ca4f5e82994dad2aa84c4e561648f298dc953e
here=$(cd "$(dirname "$0")" && pwd)
work="$here/.engine"
if [ ! -d "$work/.git" ]; then git clone -q "$ENGINE_REPO" "$work"; fi
git -C "$work" fetch -q origin "$ENGINE_REF" || true
git -C "$work" checkout -q "$ENGINE_REF"
# まいにち で選べるモデルを足す（LFM2.5 1.2B JP）。決まったコミットの catalog に、ここで書き足す
git -C "$work" checkout -q -- packages/model-manager/src/catalog.ts
python3 - "$work/packages/model-manager/src/catalog.ts" "$here/extra-models.ts.txt" <<'PY'
import sys
p, extra = sys.argv[1], open(sys.argv[2]).read()
s = open(p).read(); key = "const ONNX: ModelDefinition[] = [\n"
assert key in s; s = s.replace(key, key + extra, 1); open(p, "w").write(s)
PY
# モデルを iPhone アプリの中で動かすバックエンド（native）は、WebGPU やブラウザのメモリの見積もりと関係がないので、互換チェックを通す
git -C "$work" checkout -q -- packages/model-manager/src/compatibility.ts
python3 - "$work/packages/model-manager/src/compatibility.ts" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
key = "  const issues: CompatibilityIssue[] = [];\n"
assert s.count(key) == 1
s = s.replace(key, "  // まいにち: runs inside ひび手帳's iPhone app (llama.cpp / Apple Intelligence), not in the browser\n  if (model.backend === \"native\") return { modelId: model.id, loadable: true, issues: [] };\n" + key, 1)
open(p, "w").write(s)
PY
rm -rf "$work/apps/mainichi" && mkdir -p "$work/apps/mainichi" && cp -R "$here"/app/* "$work/apps/mainichi/"
cd "$work" && pnpm install --frozen-lockfile=false --silent
cd "$work/apps/mainichi" && MAINICHI_OUT="$here/../ai" pnpm exec vite build
