# vcpkg overlay ports

These ports override the versions from the registry baseline for vcpkg
manifest builds. They are registered in the repository's
`vcpkg-configuration.json`, so they apply to both CI and local builds.

## METIS and GKlib

The newer METIS/GKlib pair in the pinned vcpkg baseline crashes in
`METIS_PartGraphKway` on Windows. The older versions used here are known to
work on Windows, but selecting them with `overrides` in `vcpkg.json` is not
sufficient: the old METIS port enables `-march=native`, which can produce
AVX-512 instructions when a package is built on one machine and then cause an
illegal-instruction failure when the binary cache restores it on another.

The overlay keeps the known-good METIS/GKlib source revisions and patches
their build scripts to use the compiler flags supplied by the vcpkg triplet.
This makes the packages portable while preserving the Windows behavior.

vcpkg overlay ports replace a complete registry port; they cannot add a patch
to an existing port. Consequently, the port manifests, portfiles, and upstream
vcpkg patches are copied here as a unit. The `overrides` entries would be
redundant because overlay ports take precedence during version resolution.

Remove these overlays after a newer METIS/GKlib pair passes the Windows graph
cut and scene clustering tests and produces portable binaries on Linux.

## OpenBLAS and LAPACK (64-bit Windows DLL builds)

The registry's `lapack` metaport uses `lapack-reference` on x64 Windows, which
is Fortran: vcpkg builds it with MinGW's gfortran, so `liblapack.dll` needs
`libgfortran-5.dll`, `libquadmath-0.dll` (LGPL-2.1), `libgcc_s_seh-1.dll` and
`libwinpthread-1.dll` beside it. On x64 Windows DLL builds (MSVC), these
overlays instead build OpenBLAS with its own LAPACK from the C translation it
ships (`C_LAPACK`, `NOFORTRAN`), and make `find_package(LAPACK)` find
OpenBLAS (`BLA_VENDOR=OpenBLAS`). The LAPACK routines are the same reference
code; no Fortran runtime is needed. Other platforms are unchanged.

Both are copies of the ports at the baseline in `vcpkg-configuration.json`
(openblas 0.3.33, lapack 2023-06-10), with the changes marked "Overlay
change". Re-copy them if the baseline moves.
