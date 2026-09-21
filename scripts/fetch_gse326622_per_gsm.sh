#!/usr/bin/env bash
# GSE326622 per-GSM fetcher.
#
# The series-level GSE326622_RAW.tar (1,132,769,280 bytes) is NOT retrievable:
# two independent attempts both died at ~780 MB with "transfer closed with
# ~352 MB remaining" (780,318,127 then 781,318,127 bytes). That is a
# reproducible server-side truncation, so retrying the tar is futile. The
# per-sample files under geo/samples/ carry identical content and the largest
# is ~187 MB, well under the cliff.
set -u
ROOT=data/raw/GSE326622
DEST="$ROOT/per_gsm"
LIST="$ROOT/filelist.txt"
MAX_TRIES=8
mkdir -p "$DEST"

curl -sS -o "$LIST" \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE326nnn/GSE326622/suppl/filelist.txt"

manifest () { awk -F'\t' '$1=="File"{print $2"\t"$4}' "$LIST"; }

manifest | while IFS=$'\t' read -r name want; do
  gsm=${name%%_*}
  pre=$(echo "$gsm" | sed -E 's/GSM([0-9]+)[0-9]{3}$/GSM\1nnn/')
  url="https://ftp.ncbi.nlm.nih.gov/geo/samples/$pre/$gsm/suppl/$name"
  dst="$DEST/$name"
  if [ -f "$dst" ] && [ "$(stat -c%s "$dst")" -eq "$want" ]; then
    echo "[$(date +%T)] have  $name"
    continue
  fi
  for try in $(seq 1 $MAX_TRIES); do
    rm -f "$dst.part"
    curl -sS --max-time 1800 --speed-limit 20000 --speed-time 120 \
         -o "$dst.part" "$url" || true
    have=0; [ -f "$dst.part" ] && have=$(stat -c%s "$dst.part")
    if [ "$have" -eq "$want" ]; then
      mv "$dst.part" "$dst"
      echo "[$(date +%T)] OK    $name ($have bytes, try $try)"
      break
    fi
    echo "[$(date +%T)] short $name $have/$want (try $try)"
    sleep 8
  done
  rm -f "$dst.part"
done

echo "[$(date +%T)] --- verify ---"
ok=0; bad=0
while IFS=$'\t' read -r name want; do
  have=0; [ -f "$DEST/$name" ] && have=$(stat -c%s "$DEST/$name")
  if [ "$have" -eq "$want" ]; then
    ok=$((ok + 1))
  else
    bad=$((bad + 1)); echo "INCOMPLETE $name $have/$want"
  fi
done < <(manifest)
echo "[$(date +%T)] complete=$ok incomplete=$bad"
