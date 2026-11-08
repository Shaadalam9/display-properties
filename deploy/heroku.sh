#!/usr/bin/env bash
# Deploy the study to Heroku, as in bazilinskyy/ehmi-vehicles, but with the video clips:
# the committed code (HEAD) plus only the clips the design needs and the ranking previews,
# pushed straight to the Heroku app's own git. The clips never go to GitHub (third-party
# footage); Heroku serves them from public/stimuli/.
#
#   deploy/heroku.sh <heroku app name>
#   DRY_RUN=1 deploy/heroku.sh <app>     build and report the size, without pushing
#
# Needs: the Heroku CLI, logged in (heroku login), and the app's CONNECTION config variable
# set to the MongoDB connection string (heroku config:set CONNECTION='...' -a <app>).
set -euo pipefail

APP="${1:?usage: deploy/heroku.sh <heroku app name>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ -n "$(git status --porcelain -- . ':!data' ':!public/stimuli')" ]; then
    echo "Uncommitted changes: commit first, the deploy uses HEAD." >&2
    [ -n "${DRY_RUN:-}" ] || exit 1
fi

# the clips the committed public/design.json asks for, plus the previews
CLIPS=$(git show HEAD:public/design.json | uv run python -c "
import json, sys
from make_design import required_clips
design = json.load(sys.stdin)
names = sorted(set(required_clips(design)) | {'stimuli/' + p['preview'] for p in design['properties'] if p.get('preview')})
print('\n'.join(names))")
MISSING=0
for clip in $CLIPS; do
    [ -f "public/$clip" ] || { echo "missing: public/$clip" >&2; MISSING=1; }
done
[ "$MISSING" = 0 ] || { echo "Render the missing clips first (render_stimuli.py)." >&2; exit 1; }

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
git archive HEAD | tar -x -C "$BUILD"
# a Node.js app only: without the Python project files Heroku does not take it for a Python app
rm -rf "$BUILD/pyproject.toml" "$BUILD/uv.lock" "$BUILD/.python-version" "$BUILD/docs" "$BUILD/tests"
for clip in $CLIPS; do
    mkdir -p "$BUILD/public/$(dirname "$clip")"
    cp "public/$clip" "$BUILD/public/$clip"
done
echo "$(echo "$CLIPS" | wc -l | tr -d ' ') clips, $(du -sh "$BUILD/public/stimuli" | cut -f1)"

echo "build: $(du -sh "$BUILD" | cut -f1) (Heroku allows 500 MB)"
if [ -n "${DRY_RUN:-}" ]; then
    echo "DRY_RUN: not pushed"
    exit 0
fi

cd "$BUILD"
git init -q -b main
git add -A
git add -f public/stimuli
git -c user.name="deploy" -c user.email="deploy@localhost" commit -q -m "Deploy $(git -C "$ROOT" rev-parse --short HEAD) with stimuli"
git push --force "https://git.heroku.com/$APP.git" main
echo "Deployed to https://$APP.herokuapp.com (open with ?debug=true to try it)"
