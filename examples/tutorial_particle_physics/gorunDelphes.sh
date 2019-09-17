#/bin/bash   
script_arg="$@"
cd /srv/madminer/examples/tutorial_particle_physics && python 2b_delphes_level_analysis.py ${script_arg}
