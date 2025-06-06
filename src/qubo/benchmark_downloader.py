#!/usr/bin/env python3
'''
Taken from https://github.com/rliang/qubo-benchmark-instances/blob/main/get.py
'''
from typing import TextIO
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from math import floor
import gzip
import multiprocessing
import numpy as np
import argparse
import os

def orlib(n: int, output_dir: str):
    gz = n >= 1000
    name = f"bqp{n}.{'gz' if gz else 'txt'}"
    print(name)
    
    orlib_dir = os.path.join(output_dir, 'orlib')
    os.makedirs(orlib_dir, exist_ok=True)

    def process(infile: TextIO):
        num = int(next(infile))
        for index in range(num):
            filepath = os.path.join(orlib_dir, f"bqp{n}.{index + 1}")
            with open(filepath, "w") as outfile:
                _, nonzeros = map(int, next(infile).split())
                print(n, file=outfile)
                for i in range(nonzeros):
                    j, i, q = map(int, next(infile).split())
                    print(i - 1, j - 1, -q * (2 if j != i else 1), file=outfile)

    with urlopen(f"http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/{name}") as infile:
        if gz:
            with gzip.open(infile, mode="rt") as ingzip:
                process(ingzip)
        else:
            process(infile)


def gka(kind: str, index: int, output_dir: str):
    print(f"gka{index}{kind}")
    gka_dir = os.path.join(output_dir, 'gka')
    os.makedirs(gka_dir, exist_ok=True)
    filepath = os.path.join(gka_dir, f"gka{index}{kind}")
    with urlopen(f"http://biqmac.uni-klu.ac.at/library/biq/gka/gka{index}{kind}.sparse") as infile:
        with open(filepath, "w") as outfile:
            n, nonzeros = map(int, next(infile).split())
            print(n, file=outfile)
            for i in range(nonzeros):
                j, i, q = map(int, next(infile).split())
                print(i - 1, j - 1, q * (2 if j != i else 1), file=outfile)


def palubeckis(n: int, index: int, density: int, seed: int, output_dir: str):
    coef = float(2048 * 1024 * 1024 - 1)
    seed = float(seed)

    def random(seed: float):
        rd = seed * 16807
        seed = rd - floor(rd / coef) * coef
        return seed, seed / (coef + 1)

    print(f"p{n}.{index}")
    palubeckis_dir = os.path.join(output_dir, 'palubeckis')
    os.makedirs(palubeckis_dir, exist_ok=True)
    filepath = os.path.join(palubeckis_dir, f"p{n}.{index}")
    with open(filepath, mode="w") as outfile:
        print(n, file=outfile)
        for i in range(n):
            seed, r = random(seed)
            print(i, i, -int(floor(r * 201.0 - 100.0)), file=outfile)
            for j in range(i + 1, n):
                seed, fl = random(seed)
                if fl * 100 <= density:
                    seed, r = random(seed)
                    print(j, i, -int(floor(r * 201.0 - 100.0)) * 2, file=outfile)


def stanford(index: int, output_dir: str):
    print(f"G{index}")
    stanford_dir = os.path.join(output_dir, 'stanford')
    os.makedirs(stanford_dir, exist_ok=True)
    filepath = os.path.join(stanford_dir, f"G{index}")
    with urlopen(f"https://web.stanford.edu/~yyye/yyye/Gset/G{index}") as infile:
        n, nonzeros = map(int, next(infile).split())
        diag = [0] * n
        with open(filepath, mode="w") as outfile:
            print(n, file=outfile)
            for _ in range(nonzeros):
                j, i, q = map(int, next(infile).split())
                diag[i - 1] -= q
                diag[j - 1] -= q
                print(i - 1, j - 1, q * 2, file=outfile)
            for i, q in enumerate(diag):
                print(i, i, q, file=outfile)


def optsicom(output_dir: str):
    print(f"set2.zip")
    optsicom_dir = os.path.join(output_dir, 'optsicom')
    os.makedirs(optsicom_dir, exist_ok=True)
    with urlopen(f"http://grafo.etsii.urjc.es/optsicom/maxcut/set2.zip") as inzip:
        with ZipFile(BytesIO(inzip.read())) as inzipfile:
            for name in inzipfile.namelist():
                with inzipfile.open(name) as infile:
                    n, nonzeros = map(int, next(infile).split())
                    diag = [0] * n
                    filepath = os.path.join(optsicom_dir, name.split(".")[0])
                    with open(filepath, mode="w") as outfile:
                        print(n, file=outfile)
                        for _ in range(nonzeros):
                            j, i, q = map(int, next(infile).split())
                            diag[i - 1] -= q
                            diag[j - 1] -= q
                            print(i - 1, j - 1, q * 2, file=outfile)
                        for i, q in enumerate(diag):
                            print(i, i, q, file=outfile)


def dimacs(index: int, output_dir: str):
    print(f"torus{index}")
    dimacs_dir = os.path.join(output_dir, 'dimacs')
    os.makedirs(dimacs_dir, exist_ok=True)
    filepath = os.path.join(dimacs_dir, f"torus{index}")
    with urlopen(f"http://dimacs.rutgers.edu/archive/Challenges/Seventh/Instances/TORUS/torus{index}.dat.gz") as infile:
        with gzip.open(infile, mode="rt") as ingzip:
            n, nonzeros = map(int, next(ingzip).split())
            diag = [0] * n
            with open(filepath, mode="w") as outfile:
                print(n, file=outfile)
                for _ in range(nonzeros):
                    j, i, q = map(int, next(ingzip).split())
                    diag[i - 1] -= q
                    diag[j - 1] -= q
                    print(i - 1, j - 1, q * 2, file=outfile)
                for i, q in enumerate(diag):
                    print(i, i, q, file=outfile)



def load_qubo_as_symmetric(filepath: str) -> np.ndarray:
    """
    Load a QUBO file (lower‐triangular, full‐coefficient convention)
    into a *symmetric* Q so that x @ Q @ x = diag + 2*offdiag exactly
    matches the original.

    File format:
      - First line: n
      - Each subsequent line: i j q_val  (with 0 <= j <= i < n),
        where q_val is the full coefficient of xi*xj.
    """
    with open(filepath, 'r') as f:
        first = f.readline().strip()
        if not first:
            raise ValueError("Empty file or missing size line.")
        n = int(first)
        Q = np.zeros((n, n), dtype=float)

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Expected 'i j q_val'; got: {line!r}")
            i, j, q_val = parts
            i = int(i)
            j = int(j)
            v = float(q_val)

            if not (0 <= i < n and 0 <= j < n):
                raise IndexError(f"Index out of range: {line!r}")

            if i == j:
                # Diagonal: store exactly
                Q[i, i] = v
            else:
                # Off‐diagonal: split v in half
                half = v / 2.0
                Q[i, j] = half
                Q[j, i] = half

    return Q

def main():
    parser = argparse.ArgumentParser(description="Download QUBO benchmark instances.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The base directory to save the downloaded benchmark files."
    )
    args = parser.parse_args()

    p = multiprocessing.Pool()
    for n in [50, 100, 250, 500, 1000, 2500]:
        p.apply_async(orlib, [n, args.output_dir])
    for kind, num in [("a", 8), ("b", 10), ("c", 7), ("d", 10), ("e", 5), ("f", 5)]:
        for i in range(num):
            p.apply_async(gka, [kind, i + 1, args.output_dir])
    for n, density_seed in [
        (3000, [(50, 31000), (80, 32000), (80, 33000), (100, 34000), (100, 35000), (100, 36000)]),
        (4000, [(50, 41000), (80, 42000), (80, 43000), (100, 44000), (100, 45000), (100, 46000)]),
        (5000, [(50, 51000), (80, 52000), (80, 53000), (100, 54000), (100, 55000), (100, 56000)]),
        (6000, [(50, 61000), (80, 62000), (100, 64000)]),
        (7000, [(50, 71000), (80, 72000), (100, 74000)]),
    ]:
        for i, (density, seed) in enumerate(density_seed, 1):
            p.apply_async(palubeckis, [n, i, density, seed, args.output_dir])
    for i in [*range(1, 68), 70, 72, 77, 81]:
        p.apply_async(stanford, [i, args.output_dir])
    p.apply_async(optsicom, [args.output_dir])
    for i in ["g3-8", "g3-15", "pm3-8-50", "pm3-15-50"]:
        p.apply_async(dimacs, [i, args.output_dir])
    p.close()
    p.join()

if __name__ == '__main__':
    main()