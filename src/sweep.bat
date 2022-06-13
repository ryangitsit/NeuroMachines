ECHO OFF

FOR %%z IN (STDP) DO (
    FOR %%y IN (1.5) DO (
        FOR %%x IN (1.5) DO (
            FOR %%w IN (500,700,1000) DO (
                FOR %%v IN (0.05, .1, .3) DO (
                    py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir dense_sweep --length 700
                )
            )
        )
    )
)


PAUSE


@REM ECHO OFF

@REM FOR %%z IN (Maass,STDP,STSP) DO (
@REM     FOR %%y IN (0, 1.5) DO (
@REM         FOR %%x IN (0.0, 1.5) DO (
@REM             FOR %%w IN (100) DO (
@REM                 FOR %%v IN (.01, 0.05, .1) DO (
@REM                     py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --rndp %%v --res_sparsity %%v --topology rnd  --dir sparse_sweep --length 700
@REM                 )
@REM                 FOR %%u IN (0, .33, .66) DO (
@REM                     py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --beta %%u --topology smw   --dir sparse_sweep --length 700
@REM                 )
@REM                 py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 10 5 2 --topology geo  --dir sparse_sweep --length 700
@REM                 py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 5 5 4 --topology geo  --dir sparse_sweep --length 700
@REM                 py -3.8 ./main.py --learning %%z --refractory %%y --delay %%x --neurons %%w --dims 100 1 1 --topology geo  --dir sparse_sweep --length 700
@REM             )
@REM         )
@REM     )
@REM )


@REM PAUSE
