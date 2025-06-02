from qiskit.primitives.containers import DataBin
from qiskit.primitives import BasePrimitiveJob
from qiskit_ibm_runtime import SamplerV2

class SamplerV2Adapter:
    """Adapter to make SamplerV2 compatible with legacy code."""
    def __init__(self, sampler: SamplerV2):
        self._sampler = sampler

    def run(self, circuits, parameter_values, **kwargs):
        # Get shots from SamplerOptions (default=1024)
        shots = int(self._sampler.options.default_shots)
        
        # Convert parameter values to PUB-compatible format
        pubs = []
        for circuit, params in zip(circuits, parameter_values):
            param_dict = {
                param: params[i]
                for i, param in enumerate(circuit.parameters)
            }
            # Correct PUB structure: (circuit, parameter_values, shots)
            pubs.append((circuit, param_dict, shots))  # No observables needed
        
        # Submit job with PUBs
        job = self._sampler.run(pubs)
        return BasePrimitiveJob(self._adapt_result(job))

    def _adapt_result(self, job):
        """Convert SamplerV2 result to legacy format."""
        result = job.result()
        return DataBin(
            quasi_dists=result.quasi_dists,
            metadata=[{"shots": md.get("shots", 1024)} for md in result.metadata]
        )