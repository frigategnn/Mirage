#!/usr/bin/bash
score="$1"
filename="$2"
nlines="${3:-101}"
if [ "${score}" = "auc" ]
then
  tail -n$nlines "${filename}" | awk -F',' '{print $2,$4,$8,$12}'
elif [ "${score}" = "acc" ]
then
  tail -n$nlines "${filename}" | awk -F',' '{print $2,$5,$9,$13}'
elif [ "${score}" = "f1" ]
then
  tail -n$nlines "${filename}" | awk -F',' '{print $2,$6,$10,$14}'
fi
