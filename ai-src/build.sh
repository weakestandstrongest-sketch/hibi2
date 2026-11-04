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
rm -rf "$work/apps/mainichi" && mkdir -p "$work/apps/mainichi" && cp "$here"/app/* "$work/apps/mainichi/"
cd "$work" && pnpm install --frozen-lockfile=false --silent
cd "$work/apps/mainichi" && MAINICHI_OUT="$here/../ai" pnpm exec vite build
