import subprocess
from datetime import datetime as dt
import logging
from .csim import run_csim 

logger = logging.getLogger(__name__)


class CsynthTimeoutError(Exception):
    """Raised when Vitis HLS times out."""
    pass


def run_csynth(filepath, kernel_name):
    csynth_elapse_time = None
    csynth_begin = dt.now()
    with open("vitis_config.cfg", "w") as file:
        file.write(
            f"""
        part=xazu1eg-sbva484-1-i

        [hls]
        flow_target=vivado
        package.output.format=ip_catalog
        package.output.syn=false
        syn.top={kernel_name}
        syn.file={filepath}/{kernel_name}.cpp
        """
        )
    try:
        result = subprocess.run(
            [
                "v++",
                "-c",
                "--mode",
                "hls",
                "--config",
                "vitis_config.cfg",
                "--work_dir",
                filepath,
            ], 
            # v++ -c --mode hls --config vitis_config.cfg --work_dir ./results/solutions/gpt-4o/...
            timeout=120,
            capture_output=True, 
            text=True
        )
        csynth_end = dt.now()
        csynth_elapse_time = csynth_end - csynth_begin
        return result, csynth_elapse_time
    except subprocess.TimeoutExpired:
        raise CsynthTimeoutError("Vitis tools stalled during HLS Synthesis.")
