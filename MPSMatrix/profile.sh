#profile.sh

# list all tempaltes
# xctrace list templates
# == Standard Templates ==
# FL-gpu-counters
# == User Templates ==
# Activity Monitor

xctrace record --template "FL-gpu-counters" \
--output my_output.trace \
--launch -- \
./mpsmm-fp16 4000 4000 4000


# to export to txt
# xctrace export --input my_output.trace --output summary.json --toc