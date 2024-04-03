
# Table of Contents

1.  [Running the code](#org8ab739a)
2.  [Dependencies](#orgd6b9c53)

This is the official repository of the code corresponding to the paper titled
[Mirage: Model-agnostic Graph Distillation for Graph Classification](https://openreview.net/forum?id=78iGZdqxYY)
accepted in the Twelfth International Conference on Learning
Representations (ICLR 2024).


<a id="org8ab739a"></a>

# Running the code

-   Mirage was evaluated on DD, IMDB-B, NCI1, ogbg-molbace, ogbg-molbbbp,
    and ogbg-molhiv.
-   To run Mirage on these corresponding datasets,
    navigate to their respective directories.
-   Mirage is implemented in two
    ways in these repositories, (a) fully in Python and (b) partially in
    C++.
-   The DD, ogbg-molbace, ogbg-molhiv implementations are of type (b)
    while the rest are of type (a).
-   To run Mirage using (a) implementation, navigate to the
    directory. The directory contains a single python script. Just run it.
-   To run Mirage using implementation (b) first complete the dependency
    installation described at [2](#orgd306116), navigate to the
    directory. While being in the main directory run `bash
      scripts/run_<dataset>.sh`. This will produce outputs in the
    `outputs` directory.


<a id="orgd6b9c53"></a>

# Dependencies

-   Running Mirage requires `NumPy`, `SciPy`, `PyTorch Geometric`, `PyTorch`,
    `scikit-learn`, `pyfpgrowth`, `Matplotlib`, and `tqdm` with `Python=3.9` on Linux
    operating systems.
-   <a id="orgd306116"></a>Specifically for implementation (b):
    1.  You will need to ensure that `1.11<=PyTorch<=1.13` is installed.
    2.  The C++ implementation is written as a [PyTorch extension](https://docs.w3cub.com/pytorch/cpp_extension) so that
        it can be called inside the Python file. Within each dataset's
        corresponding folder that has a C++ implementation, this PyTorch
        extension is present inside directory called `pygcanl`.
    3.  To work with `pygcanl`, you need to install it. Run the following
        command: `$ pip install -e pygcanl`.
    4.  If this runs without errors, then that's it, you can run Mirage now.

