find . -type f -size -3M \
! -name "*.wav" \
! -name "*.flac" \
| xargs git add