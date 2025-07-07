import subprocess
from .csim import run_csim 


class CsynthTimeoutError(Exception):
    """Raised when Vitis HLS times out."""
    pass

# Run vitis tools on the generated code
def run_vitis_tools(filepath, kernel_name, mode="csynth", hls_flow="vitis"):
    if mode == "csynth":
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
            return result
        except subprocess.TimeoutExpired:
            raise CsynthTimeoutError("Vitis tools stalled during HLS Synthesis.")
    elif mode == "csim":
        return run_csim(kernel_name, filepath, hls_flow=hls_flow)
        
