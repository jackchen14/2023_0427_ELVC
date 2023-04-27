import os
import ipdb
from pathlib import Path
from scipy.io import wavfile


NL_parse_dir = Path('NL_parse')
NL_parse_dir.mkdir(parents=True, exist_ok=True)

NL_parse_timing = open(str(NL_parse_dir/'NL_parse_timing.scp'),'w')

NL_parse_timing.close()