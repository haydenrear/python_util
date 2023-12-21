import torch


def ifft_to_real(ifft_taken_already: torch.Tensor):
    magnitude = torch.abs(ifft_taken_already)  # Magnitude of the complex numbers
    phase = torch.angle(ifft_taken_already)    # Phase angle of the complex numbers
    out = magnitude * torch.cos(phase)
    return out
