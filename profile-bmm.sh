#profile.sh

# list all tempaltes
# xctrace list templates
# == Standard Templates ==
# FL-gpu-counters
# == User Templates ==
# Activity Monitor

xctrace record --template "FL-gpu-counters" \
--output bmm.trace \
--launch -- \
/Users/felixlin/workspace-mps/myenv/bin/python3 bmm.py


# to export to txt
# xctrace export --input my_output.trace --output summary.json --toc