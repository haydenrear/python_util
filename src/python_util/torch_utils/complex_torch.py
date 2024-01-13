import torch


def ifft_to_real(ifft_taken_already: torch.Tensor):
    magnitude = torch.abs(ifft_taken_already)  # Magnitude of the complex numbers
    phase = torch.angle(ifft_taken_already)  # Phase angle of the complex numbers
    out = (magnitude + 1e-12) * torch.cos(phase)
    return out


def complex_to_2d(complex_numbered: torch.Tensor):
    reshaped = torch.view_as_real(complex_numbered)
    if len(reshaped.shape) > 2:
        reshape_size = [i for i in reshaped.shape[:-2]]
        reshape_size.append(reshaped.shape[-1] * reshaped.shape[-2])
        return reshaped.reshape(reshape_size)
    else:
        return reshaped.reshape(reshaped.shape[0] * reshaped.shape[1])


def to_complex_from_2d(to_project_to_complex: torch.Tensor):
    """
    Assumes imaginary part is first
    :param to_project_to_complex:
    :return:
    """
    shape_curr = [s for s in to_project_to_complex.shape[:-1]]
    assert to_project_to_complex.shape[-1] % 2 == 0, "To project to complex must have real and imaginary part."
    shape_curr.extend([to_project_to_complex.shape[-1] // 2, 2])
    to_project_to_complex = to_project_to_complex.reshape(shape_curr)
    return torch.view_as_complex(to_project_to_complex)
