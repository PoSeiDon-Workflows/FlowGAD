#!/usr/bin/bash env

for i in {71..80}; do rm $(find data_new -name montage-run00${i}.csv); done

for i in data_new/*; do mkdir -p bad_montage_data/${i##*/}; done

for i in $(find data_new -name montage-run*.csv); do
    if [[ "${i##*/}" < "montage-run0536.csv" ]]; then
         mv $i bad_montage_data/${i#*/}
    fi
done
