from types import SimpleNamespace

from scripts.runtime_process_timing import dimensions, params_for, summary


def test_preview_uses_the_requested_actual_long_edge():
    assert dimensions('preview640', (1000, 667, 3)) == (640, 426)
    assert dimensions('preview1280', (1000, 667, 3)) == (1280, 853)
    assert dimensions('raw', (5000, 7000, 3)) == (5000, 7000)


def test_cold_pass_excluded_from_warm_median_and_dispersion():
    samples = [{'phase': 'cold_process', 'engine_seconds': 50}]
    samples += [{'phase': 'warm', 'engine_seconds': t} for t in (2, 3, 5)]
    assert summary(samples) == {'warm_samples': 3, 'median_seconds': 3, 'mad_seconds': 1,
                                'min_seconds': 2, 'max_seconds': 5}


def test_options_switch_stock_and_mode_without_changing_halation_or_compressor():
    args = SimpleNamespace(case='6mp', change='stock', mode='alternate', grain_off=False,
                           halation_off=False, deterministic=True)
    paper, negative = (params_for(args, i) for i in (0, 1))
    assert paper.film.info.stock == 'kodak_gold_200'
    assert negative.film.info.stock == 'kodak_portra_400'
    assert not paper.io.scan_film and negative.io.scan_film
    for p in (paper, negative):
        assert p.film_render.halation.active
        assert not p.film_render.grain.active
        assert p.io.output_gamut_compress.algorithm == 'cam16ucs'
