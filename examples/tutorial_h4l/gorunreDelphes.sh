#/bin/bash   
script_arg="$@"
cd /srv/madminer/examples/tutorial_h4l && python reRunDelphes.py ${script_arg}
