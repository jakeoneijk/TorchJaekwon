find . -type f -size -1M \
! -name "*.wav" \
! -name "*.flac" \
| xargs git add