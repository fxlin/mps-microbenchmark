#profile.sh

# list all tempaltes
# xctrace list templates
# == Standard Templates ==
# FL-gpu-counters
# == User Templates ==
# Activity Monitor

xctrace record --template "FL-gpu-counters" \
--output metal_half.trace \
--launch -- \
matrix_multiplication


# to export to txt
# xctrace export --input my_output.trace --output summary.json --toc