import os


def get_allowed_n_proc_DA():
    """Return augmentation workers per GPU, using nnUNet_n_proc_DA or the default of 12."""
    if 'nnUNet_n_proc_DA' in os.environ:
        use_this = int(os.environ['nnUNet_n_proc_DA'])
    else:
        use_this = 12
    return min(use_this, os.cpu_count())
