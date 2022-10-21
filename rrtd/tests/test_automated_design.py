import automated_design

def test_parallel_compute():
    args = list(range(1000))
    for kw in [
        dict(n_jobs=1, chunk_size=1),
        dict(n_jobs=3, chunk_size=1),
        dict(n_jobs=3, chunk_size=3),
        dict(n_jobs=1, chunk_size=3),
    ]:
        rv = automated_design.parallel_compute(lambda i: i**2, args, **kw)
        assert rv == [a**2 for a in args]
