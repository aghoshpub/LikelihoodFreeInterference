#/bin/bash   
script_arg="$@"
cd /srv/madminer/examples/tutorial_h4l && python 2b_delphes_level_analysis.py ${script_arg}
