#!/usr/bin/env zsh

# validator-v3.sh
# 
# checks a directory full of SD card data for rows with invalid numbers of columns

usage() {
  echo "usage: $(basename $0) [path-to-sd-card-files]"
  echo
  echo "be sure to properly quote the path if it contains spaces or special characters"
}

case $# in 
  1) datapath=$1;;
  *) usage; exit 1;;
esac

echo "++ working with files in ${datapath}"
echo

for f in "${datapath}"/*_(FLT|LOC|SST).* 
do
   awk '
BEGIN { 
    FS = "," 
    fltcols = 6
    loccols = 5
    sstcols = 2
}
FILENAME ~ /.*FLT\.[csvCSV]+*/ && NF != fltcols && NR > 1 { printf("%s row %d = %d (expecting %d)\n", FILENAME, NR, NF, fltcols) }
FILENAME ~ /.*LOC\.[csvCSV]+*/ && NF != loccols { printf("%s row %d = %d (expecting %d)\n", FILENAME, NR, NF, loccols) }
FILENAME ~ /.*SST\.[csvCSV]+*/ && NF != sstcols { printf("%s row %d = %d (expecting %d)\n", FILENAME, NR, NF, sstcols) }
FILENAME ~ /.*SST\.[csvCSV]+*/ && $2 == "1" { printf("%s row %d = %d (bad temp)\n", FILENAME, NR, $2) }
' "${f}"
done

